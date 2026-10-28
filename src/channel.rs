//! Preview channel from a decoder job to the GUI: length-prefixed packets over a
//! loopback TCP connection, which works the same on Linux, macOS and Windows.
//! The sender never blocks. While one packet is still being written, newer
//! packets are dropped; the GUI only needs the latest picture, and RF decoding
//! is never delayed by the display.
use std::{
    io::{self, Read, Write},
    net::{Ipv4Addr, TcpListener, TcpStream},
};

/// Largest accepted packet: a full-raster NV12 colour field is about 535 KB.
const MAX_PACKET: usize = 4 * 1024 * 1024;

pub struct PreviewSender {
    socket: TcpStream,
    pending: Vec<u8>,
    offset: usize,
}
impl PreviewSender {
    pub fn connect(port: u16) -> io::Result<Self> {
        let socket = TcpStream::connect((Ipv4Addr::LOCALHOST, port))?;
        socket.set_nodelay(true)?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket,
            pending: Vec::new(),
            offset: 0,
        })
    }
    /// Queue a packet; `Ok(false)` when it was dropped because the previous one
    /// is still being written.
    pub fn send(&mut self, packet: &[u8]) -> io::Result<bool> {
        self.flush()?;
        if self.offset < self.pending.len() {
            return Ok(false);
        }
        self.pending.clear();
        self.offset = 0;
        self.pending
            .extend_from_slice(&(packet.len() as u32).to_le_bytes());
        self.pending.extend_from_slice(packet);
        self.flush()?;
        Ok(true)
    }
    fn flush(&mut self) -> io::Result<()> {
        while self.offset < self.pending.len() {
            match self.socket.write(&self.pending[self.offset..]) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(count) => self.offset += count,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

pub struct PreviewReceiver {
    listener: TcpListener,
    socket: Option<TcpStream>,
    buffer: Vec<u8>,
}
impl PreviewReceiver {
    /// Listen on a free loopback port for one decoder connection.
    pub fn bind() -> io::Result<Self> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        listener.set_nonblocking(true)?;
        Ok(Self {
            listener,
            socket: None,
            buffer: Vec::new(),
        })
    }
    pub fn port(&self) -> io::Result<u16> {
        Ok(self.listener.local_addr()?.port())
    }
    /// Complete packets received since the last call, oldest first.
    pub fn receive(&mut self) -> io::Result<Vec<Vec<u8>>> {
        if self.socket.is_none() {
            match self.listener.accept() {
                Ok((socket, _)) => {
                    socket.set_nonblocking(true)?;
                    self.socket = Some(socket);
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(Vec::new()),
                Err(e) => return Err(e),
            }
        }
        let socket = self.socket.as_mut().unwrap();
        let mut chunk = [0u8; 65536];
        loop {
            match socket.read(&mut chunk) {
                Ok(0) => break,
                Ok(count) => self.buffer.extend_from_slice(&chunk[..count]),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        let mut packets = Vec::new();
        let mut at = 0;
        while self.buffer.len() - at >= 4 {
            let length = u32::from_le_bytes(self.buffer[at..at + 4].try_into().unwrap()) as usize;
            if length > MAX_PACKET {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Preview packet is too large",
                ));
            }
            if self.buffer.len() - at - 4 < length {
                break;
            }
            packets.push(self.buffer[at + 4..at + 4 + length].to_vec());
            at += 4 + length;
        }
        self.buffer.drain(..at);
        Ok(packets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packets_arrive_whole_and_in_order() {
        let mut receiver = PreviewReceiver::bind().unwrap();
        let mut sender = PreviewSender::connect(receiver.port().unwrap()).unwrap();
        let mut received = Vec::new();
        for value in 0..3u8 {
            // Packets larger than a socket buffer are split and reassembled.
            let packet = vec![value; 300_000];
            while !sender.send(&packet).unwrap() {
                received.extend(receiver.receive().unwrap());
            }
        }
        for _ in 0..1000 {
            received.extend(receiver.receive().unwrap());
            sender.send(&[]).ok();
            if received.iter().filter(|p| !p.is_empty()).count() == 3 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let data: Vec<_> = received.iter().filter(|p| !p.is_empty()).collect();
        assert_eq!(data.len(), 3);
        for (value, packet) in data.iter().enumerate() {
            assert!(packet.len() == 300_000 && packet.iter().all(|&b| b == value as u8));
        }
    }
}
