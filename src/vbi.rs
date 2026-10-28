//! Raw PAL VBI transport: 16 lines × 1135 UInt16 LE samples per field.
//!
//! The receiver may send teletext header images back on the same connection.
//! Each message is a 20-byte little-endian header followed by RGBA pixels:
//! `b"OXTH"`, version 1, kind 1 (header row image), 2 reserved bytes, the VBI
//! frame index counted from the first field sent on this connection (two fields
//! per frame, i64), width and height (u16 each), then width × height × 4 bytes
//! of straight-alpha RGBA.
use std::{
    collections::VecDeque,
    io::{self, Read, Write},
    net::{TcpListener, TcpStream},
    thread,
    time::Duration,
};

pub const WIDTH: usize = 1135;
pub const FIRST_ROW: usize = 6;
pub const LINES: usize = 16;
pub const FIELD_BYTES: usize = WIDTH * LINES * 2;
const MAX_FIELDS: usize = 64;
const MESSAGE_HEADER: usize = 20;
const MAX_IMAGE_WIDTH: usize = 2048;
const MAX_IMAGE_HEIGHT: usize = 256;
/// Decoder field numbers of the last fields sent: a minute of PAL fields, so
/// a slow receiver's images still map to the field they came from.
const FIELD_HISTORY: usize = 3000;

/// A teletext header row rendered by the receiver, tied to the decoder field
/// (output sequence number) of the first field of its VBI frame.
#[derive(Clone)]
pub struct HeaderImage {
    pub field: u64,
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

pub struct Server {
    listener: TcpListener,
    client: Option<TcpStream>,
    queue: VecDeque<Vec<u8>>,
    offset: usize,
    previous: Option<bool>,
    sent: u64,
    /// Decoder field number of each field queued on this connection, by index.
    fields: VecDeque<u64>,
    first_field_index: u64,
    received: Vec<u8>,
    headers: Vec<HeaderImage>,
}
impl Server {
    pub fn bind(address: &str) -> Result<Self, String> {
        let listener = TcpListener::bind(address).map_err(|e| format!("VBI listener: {e}"))?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        println!("VBI_SERVER listening={} format=u16le samples_per_line={WIDTH} lines_per_field={LINES} first_tbc_row={FIRST_ROW} header=none", listener.local_addr().map_err(|e| e.to_string())?);
        Ok(Self {
            listener,
            client: None,
            queue: VecDeque::new(),
            offset: 0,
            previous: None,
            sent: 0,
            fields: VecDeque::new(),
            first_field_index: 0,
            received: Vec::new(),
            headers: Vec::new(),
        })
    }
    fn accept(&mut self) -> Result<(), String> {
        if self.client.is_none() {
            match self.listener.accept() {
                Ok((socket, peer)) => {
                    socket.set_nonblocking(true).map_err(|e| e.to_string())?;
                    socket.set_nodelay(true).map_err(|e| e.to_string())?;
                    self.client = Some(socket);
                    self.previous = None;
                    self.fields.clear();
                    self.first_field_index = 0;
                    self.received.clear();
                    println!("VBI_SERVER connected={peer} waiting_for_first_field=true");
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(format!("VBI accept: {e}")),
            }
        }
        Ok(())
    }
    pub fn wait_for_client(&mut self) -> Result<(), String> {
        println!("VBI_SERVER waiting_for_client=true");
        while self.client.is_none() && !crate::native::stop_requested() {
            self.accept()?;
            if self.client.is_none() {
                thread::sleep(Duration::from_millis(25));
            }
        }
        Ok(())
    }
    fn disconnect(&mut self, reason: &str) {
        eprintln!("VBI_SERVER disconnected: {reason}; decoding continues, reconnect starts at a first field");
        self.client = None;
        self.queue.clear();
        self.offset = 0;
        self.previous = None;
        self.fields.clear();
        self.received.clear();
    }
    /// Read and parse header images sent back by the receiver.
    fn receive(&mut self) {
        let mut buffer = [0u8; 65536];
        while let Some(socket) = &mut self.client {
            match socket.read(&mut buffer) {
                Ok(0) => {
                    self.disconnect("peer closed");
                    return;
                }
                Ok(count) => self.received.extend_from_slice(&buffer[..count]),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) => {
                    self.disconnect(&e.to_string());
                    return;
                }
            }
        }
        while self.received.len() >= MESSAGE_HEADER {
            let header = &self.received[..MESSAGE_HEADER];
            let width = u16::from_le_bytes([header[16], header[17]]) as usize;
            let height = u16::from_le_bytes([header[18], header[19]]) as usize;
            if &header[..4] != b"OXTH"
                || header[4] != 1
                || header[5] != 1
                || !(1..=MAX_IMAGE_WIDTH).contains(&width)
                || !(1..=MAX_IMAGE_HEIGHT).contains(&height)
            {
                self.disconnect("invalid teletext header message from receiver");
                return;
            }
            let length = MESSAGE_HEADER + width * height * 4;
            if self.received.len() < length {
                break;
            }
            let frame = i64::from_le_bytes(header[8..16].try_into().unwrap());
            let rgba = self.received[MESSAGE_HEADER..length].to_vec();
            self.received.drain(..length);
            // VBI frame n consists of fields 2n and 2n + 1 sent on this connection.
            let index = u64::try_from(frame).ok().and_then(|f| f.checked_mul(2));
            let field = index
                .and_then(|i| i.checked_sub(self.first_field_index))
                .and_then(|i| self.fields.get(i as usize).copied());
            if let Some(field) = field {
                self.headers.push(HeaderImage {
                    field,
                    width,
                    height,
                    rgba,
                });
            }
        }
    }
    /// Header images received since the last call, oldest first.
    pub fn take_headers(&mut self) -> Vec<HeaderImage> {
        self.accept().ok();
        self.receive();
        std::mem::take(&mut self.headers)
    }
    fn flush(&mut self) {
        while let (Some(socket), Some(field)) = (&mut self.client, self.queue.front()) {
            match socket.write(&field[self.offset..]) {
                Ok(0) => {
                    self.disconnect("peer closed");
                    break;
                }
                Ok(count) => {
                    self.offset += count;
                    if self.offset == field.len() {
                        self.queue.pop_front();
                        self.offset = 0;
                        self.sent += 1;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) => {
                    self.disconnect(&e.to_string());
                    break;
                }
            }
        }
    }
    /// Samples are already resampled/levelled TBC luma, not display pixels.
    /// `field` is the decoder's output sequence number of this field.
    pub fn publish(&mut self, samples: &[u16], first: bool, field: u64) -> Result<(), String> {
        if samples.len() != WIDTH * LINES {
            return Err("Invalid VBI field dimensions".into());
        }
        self.accept()?;
        self.flush();
        if self.client.is_none() || (self.previous.is_none() && !first) {
            return Ok(());
        }
        if self.previous == Some(first) {
            self.disconnect("field polarity discontinuity");
            return Ok(());
        }
        if self.queue.len() >= MAX_FIELDS {
            self.disconnect("slow receiver exceeded the bounded VBI buffer");
            return Ok(());
        }
        self.queue
            .push_back(samples.iter().flat_map(|x| x.to_le_bytes()).collect());
        self.previous = Some(first);
        self.fields.push_back(field);
        if self.fields.len() > FIELD_HISTORY {
            self.fields.pop_front();
            self.first_field_index += 1;
        }
        self.flush();
        Ok(())
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        // Drain briefly at EOF without allowing a slow consumer to stall RF.
        let deadline = std::time::Instant::now() + Duration::from_millis(500);
        while !self.queue.is_empty() && std::time::Instant::now() < deadline {
            self.flush();
            if !self.queue.is_empty() {
                thread::sleep(Duration::from_millis(5));
            }
        }
        println!(
            "VBI_SERVER sent_fields={} pending_fields={}",
            self.sent,
            self.queue.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    #[test]
    fn raw_stream_starts_at_first_field_and_preserves_u16_sample_order() {
        let mut server = Server::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(server.listener.local_addr().unwrap()).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let samples: Vec<u16> = (0..WIDTH * LINES).map(|i| (i * 31) as u16).collect();
        server.publish(&samples, false, 0).unwrap();
        assert_eq!(server.sent, 0);
        server.publish(&samples, true, 1).unwrap();
        let mut bytes = vec![0; FIELD_BYTES];
        client.read_exact(&mut bytes).unwrap();
        let expected: Vec<u8> = samples.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(
            bytes, expected,
            "No header, truncation or sample conversion is allowed"
        );
        server.publish(&samples, false, 0).unwrap();
        client.read_exact(&mut bytes).unwrap();
        assert_eq!(bytes, expected);
        server.publish(&samples, false, 0).unwrap();
        assert!(
            server.client.is_none(),
            "A polarity discontinuity ends the session"
        );
        assert!(server.publish(&[0], true, 0).is_err());
    }
    #[test]
    fn header_images_map_to_the_decoder_field_of_their_vbi_frame() {
        let mut server = Server::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(server.listener.local_addr().unwrap()).unwrap();
        let samples = vec![0u16; WIDTH * LINES];
        // A leading second field is not sent; decoder fields 11.. form VBI frames.
        server.publish(&samples, false, 10).unwrap();
        for field in 11..17 {
            server.publish(&samples, field % 2 == 1, field).unwrap();
        }
        let mut message = b"OXTH\x01\x01\0\0".to_vec();
        message.extend(2i64.to_le_bytes());
        message.extend(3u16.to_le_bytes());
        message.extend(2u16.to_le_bytes());
        message.extend((0..24).map(|i| i as u8));
        // Split delivery must be reassembled.
        client.write_all(&message[..7]).unwrap();
        thread::sleep(Duration::from_millis(20));
        assert!(server.take_headers().is_empty());
        client.write_all(&message[7..]).unwrap();
        thread::sleep(Duration::from_millis(20));
        let headers = server.take_headers();
        assert_eq!(headers.len(), 1);
        assert_eq!(
            headers[0].field, 15,
            "VBI frame 2 starts at the fifth sent field"
        );
        assert_eq!((headers[0].width, headers[0].height), (3, 2));
        assert_eq!(headers[0].rgba[23], 23);
        client.write_all(b"JUNKJUNKJUNKJUNKJUNKJUNK").unwrap();
        thread::sleep(Duration::from_millis(20));
        assert!(server.take_headers().is_empty());
        assert!(
            server.client.is_none(),
            "Malformed messages end the session"
        );
    }
}
