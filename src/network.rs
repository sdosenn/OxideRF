//! Native, lossless RF transport for a trusted LAN. One receiver per session.
use std::{
    fs::File,
    io::{self, Read, Write},
    net::{TcpListener, TcpStream, ToSocketAddrs},
    path::Path,
    thread,
    time::{Duration, Instant},
};

const MAGIC: &[u8; 8] = b"VHSRF001";
const READY: &[u8; 4] = b"RDY!";
const HEADER_SIZE: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Profile {
    pub rate: f64,
    pub bytes: usize,
    pub audio: Option<(u16, u64)>,
}
impl Profile {
    fn encode(self) -> [u8; HEADER_SIZE] {
        let mut data = [0; HEADER_SIZE];
        data[..8].copy_from_slice(if self.audio.is_some() {
            b"VHSRF002"
        } else {
            MAGIC
        });
        data[8] = self.bytes as u8;
        data[16..24].copy_from_slice(&self.rate.to_le_bytes());
        if let Some((port, token)) = self.audio {
            data[9] = 1;
            data[10..12].copy_from_slice(&port.to_le_bytes());
            data[24..].copy_from_slice(&token.to_le_bytes());
        }
        data
    }
    fn decode(data: &[u8; HEADER_SIZE]) -> Result<Self, String> {
        let audio =
            if &data[..8] == b"VHSRF002" && data[9] == 1 && data[12..16].iter().all(|v| *v == 0) {
                let port = u16::from_le_bytes(data[10..12].try_into().unwrap());
                if port == 0 {
                    return Err("Invalid server audio port".into());
                }
                Some((port, u64::from_le_bytes(data[24..].try_into().unwrap())))
            } else {
                None
            };
        if (audio.is_none() && &data[..8] != MAGIC)
            || ![1, 2].contains(&data[8])
            || (audio.is_none() && data[9..16].iter().chain(&data[24..]).any(|&v| v != 0))
        {
            return Err("Unsupported RF stream header; use the native --stream-rf sender".into());
        }
        let rate = f64::from_le_bytes(data[16..24].try_into().unwrap());
        if !(14e6..=100e6).contains(&rate) {
            return Err("RF stream sample rate must be 14–100 MHz".into());
        }
        Ok(Self {
            rate,
            bytes: data[8] as usize,
            audio,
        })
    }
}

fn read_handshake(
    socket: &mut TcpStream,
    data: &mut [u8],
    timeout: Duration,
) -> Result<(), String> {
    socket
        .set_read_timeout(Some(Duration::from_millis(100)))
        .map_err(|e| e.to_string())?;
    let deadline = Instant::now() + timeout;
    let mut received = 0;
    while received < data.len() {
        if crate::native::stop_requested() {
            return Err("RF connection cancelled".into());
        }
        if Instant::now() >= deadline {
            return Err("RF handshake timed out".into());
        }
        match socket.read(&mut data[received..]) {
            Ok(0) => return Err("RF connection closed during handshake".into()),
            Ok(n) => received += n,
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::Interrupted
                ) => {}
            Err(e) => return Err(format!("RF handshake: {e}")),
        }
    }
    socket.set_read_timeout(None).map_err(|e| e.to_string())
}

pub(crate) struct Connection {
    socket: TcpStream,
    pub profile: Profile,
    address: String,
}
impl Connection {
    pub(crate) fn connect(url: &str) -> Result<Self, String> {
        let address = url
            .strip_prefix("tcp://")
            .ok_or("Network input must be tcp://HOST:PORT")?;
        if address.is_empty() || address.contains('/') {
            return Err("Network input must be tcp://HOST:PORT".into());
        }
        let addresses = address
            .to_socket_addrs()
            .map_err(|e| format!("RF server address: {e}"))?;
        let mut socket = None;
        let mut error = "No RF server addresses found".to_string();
        for address in addresses {
            if crate::native::stop_requested() {
                return Err("RF connection cancelled".into());
            }
            match TcpStream::connect_timeout(&address, Duration::from_secs(2)) {
                Ok(stream) => {
                    socket = Some(stream);
                    break;
                }
                Err(e) => error = format!("Cannot connect to RF server {address}: {e}"),
            }
        }
        let mut socket = socket.ok_or(error)?;
        socket.set_nodelay(true).map_err(|e| e.to_string())?;
        socket
            .set_write_timeout(Some(Duration::from_millis(500)))
            .map_err(|e| e.to_string())?;
        let mut header = [0; HEADER_SIZE];
        read_handshake(&mut socket, &mut header, Duration::from_secs(5))?;
        let profile = Profile::decode(&header)?;
        println!(
            "NETWORK_RF server={address} format={} rate_mhz={:.9} required_mbit_s={:.2}",
            if profile.bytes == 1 { "u8" } else { "u16" },
            profile.rate / 1e6,
            profile.rate * profile.bytes as f64 * 8.0 / 1e6
        );
        Ok(Self {
            socket,
            profile,
            address: address.into(),
        })
    }
    pub(crate) fn audio(
        &self,
        base: Option<&Path>,
        monitor: bool,
    ) -> Result<Option<crate::audio::Receiver>, String> {
        self.profile
            .audio
            .map(|(port, token)| {
                crate::audio::Receiver::connect(
                    &self.address,
                    port,
                    token,
                    base,
                    monitor,
                    self.profile.rate,
                )
            })
            .transpose()
    }
    pub(crate) fn start(mut self) -> Result<crate::cxadc::Stream, String> {
        // The sender must not open the ADC until GPU initialization completes.
        self.socket
            .write_all(READY)
            .map_err(|e| format!("RF receiver ready: {e}"))?;
        crate::cxadc::Stream::from_tcp(self.socket, self.profile.bytes, self.profile.rate)
    }
}

pub fn serve(args: &[String]) -> Result<(), String> {
    let allowed = [
        "--input",
        "--bind",
        "--sample-format",
        "--frequency",
        "--persistent",
        "--audio-source",
        "--audio-bind",
    ];
    if !args.len().is_multiple_of(2)
        || args
            .as_chunks::<2>()
            .0
            .iter()
            .any(|p| !allowed.contains(&p[0].as_str()))
    {
        return Err("Usage: --stream-rf --input /dev/cxadc0 --bind HOST:5555 [--sample-format u8|u16] [--frequency MHz]".into());
    }
    let get = |key: &str| {
        args.as_chunks::<2>()
            .0
            .iter()
            .find(|p| p[0] == key)
            .map(|p| p[1].as_str())
    };
    let path = Path::new(get("--input").unwrap_or(crate::cxadc::DEFAULT_DEVICE));
    let device = crate::cxadc::inspect(path);
    let format = get("--sample-format")
        .or(device.sample_format())
        .ok_or("Cannot detect sender format; specify --sample-format u8 or u16")?;
    let bytes = match format {
        "u8" => 1,
        "u16" => 2,
        _ => return Err("RF sender supports u8 or u16 LE".into()),
    };
    let rate = if let Some(rate) = get("--frequency") {
        rate.parse::<f64>().map_err(|e| e.to_string())? * 1e6
    } else {
        device
            .frequency_mhz
            .ok_or("Cannot detect sender sample rate; specify --frequency MHz")?
            * 1e6
    };
    if let Some(actual) = device.sample_format() {
        if format != actual {
            return Err(format!(
                "Sender device requires {actual}, selected {format}"
            ));
        }
    }
    let mut profile = Profile {
        bytes,
        rate,
        audio: None,
    };
    Profile::decode(&profile.encode())?;
    crate::native::install_stop_handler()?;
    let listener = TcpListener::bind(get("--bind").unwrap_or("0.0.0.0:5555"))
        .map_err(|e| format!("RF listener: {e}"))?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let audio_listener = if get("--audio-source").is_some() {
        if !crate::cxadc::is_stream(path) {
            return Err("Server audio requires live cxadc/FIFO RF input".into());
        }
        let listener = TcpListener::bind(get("--audio-bind").unwrap_or("0.0.0.0:5557"))
            .map_err(|e| format!("Audio listener: {e}"))?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        Some(listener)
    } else {
        None
    };
    println!(
        "RF_SERVER listening={} input={} format={format} rate_mhz={:.9} waiting_for_receiver=true",
        listener.local_addr().map_err(|e| e.to_string())?,
        path.display(),
        rate / 1e6
    );
    let persistent = match get("--persistent").unwrap_or("false") {
        "true" => true,
        "false" => false,
        _ => return Err("Persistent mode must be true or false".into()),
    };
    loop {
        let (mut socket, peer) = loop {
            if crate::native::stop_requested() {
                return Ok(());
            }
            match listener.accept() {
                Ok(connection) => break connection,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(25))
                }
                Err(e) => return Err(format!("RF accept: {e}")),
            }
        };
        let session = (|| -> Result<(), String> {
            if let Some(listener) = &audio_listener {
                let token = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|e| e.to_string())?
                    .as_nanos() as u64
                    ^ std::process::id() as u64;
                profile.audio = Some((
                    listener.local_addr().map_err(|e| e.to_string())?.port(),
                    token,
                ));
            }
            socket
                .set_write_timeout(Some(Duration::from_millis(500)))
                .map_err(|e| e.to_string())?;
            socket
                .write_all(&profile.encode())
                .map_err(|e| e.to_string())?;
            let audio = audio_listener
                .as_ref()
                .map(|listener| {
                    crate::audio::SenderReady::accept(
                        listener,
                        profile.audio.unwrap().1,
                        get("--audio-source").unwrap(),
                    )
                })
                .transpose()?;
            let mut ready = [0; 4];
            read_handshake(&mut socket, &mut ready, Duration::from_secs(60))?;
            if &ready != READY {
                return Err("Invalid RF receiver readiness response".into());
            }
            // Files support deterministic transport tests and replay; real devices use
            // the bounded acquisition worker so network stalls cannot silently lose RF.
            let live = crate::cxadc::is_stream(path);
            let mut capture = if live {
                Some(crate::cxadc::Stream::open(path, format, rate)?)
            } else {
                None
            };
            let mut file = if live {
                None
            } else {
                Some(File::open(path).map_err(|e| e.to_string())?)
            };
            let _audio = audio
                .map(|audio| audio.start(capture.as_ref().unwrap().clock()))
                .transpose()?;
            println!("RF_SERVER connected={peer} capture_started=true");
            let mut sent = 0u64;
            let transfer =
                (|| {
                    while !crate::native::stop_requested() {
                        let data = if let Some(capture) = &mut capture {
                            capture.read_raw(256 * 1024 / bytes)?
                        } else {
                            let mut data = vec![0; 256 * 1024];
                            let n = file
                                .as_mut()
                                .unwrap()
                                .read(&mut data)
                                .map_err(|e| e.to_string())?;
                            data.truncate(n);
                            data
                        };
                        if data.is_empty() {
                            break;
                        }
                        socket.write_all(&data).map_err(|e| {
                format!("RF network send failed: {e}. Capture stopped; no automatic reconnect.")
            })?;
                        sent += data.len() as u64;
                    }
                    if !sent.is_multiple_of(bytes as u64) {
                        return Err("RF sender ended with an incomplete PCM sample".into());
                    }
                    Ok(())
                })();
            println!(
                "RF_SERVER sent_bytes={sent} stopped={}",
                crate::native::stop_requested()
            );
            transfer
        })();
        if crate::native::stop_requested() {
            return Ok(());
        }
        if !persistent {
            return session;
        }
        if let Err(error) = session {
            eprintln!("RF_SERVER session_ended: {error}");
        }
        println!("RF_SERVER waiting_for_receiver=true");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rf_header_roundtrip_rejects_unknown_formats_and_invalid_rates() {
        let profile = Profile {
            rate: 315e6 / 11.0,
            bytes: 1,
            audio: Some((5557, 123456)),
        };
        assert_eq!(Profile::decode(&profile.encode()).unwrap(), profile);
        for bytes in [1, 2] {
            let profile = Profile {
                rate: 315e6 / 11.0,
                bytes,
                audio: None,
            };
            assert_eq!(Profile::decode(&profile.encode()).unwrap(), profile);
        }
        for rate in [0.0, f64::NAN, f64::INFINITY, 101e6] {
            assert!(Profile::decode(
                &Profile {
                    rate,
                    bytes: 1,
                    audio: None
                }
                .encode()
            )
            .is_err());
        }
        let mut header = Profile {
            rate: 28e6,
            bytes: 1,
            audio: None,
        }
        .encode();
        header[8] = 4;
        assert!(Profile::decode(&header).is_err());
        header[8] = 1;
        header[9] = 1;
        assert!(Profile::decode(&header).is_err());
    }
    #[test]
    fn network_reader_preserves_split_samples_and_accepts_clean_empty_eof() {
        for bytes in [1, 2] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let mut writer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            let socket = listener.accept().unwrap().0;
            let mut reader = crate::cxadc::Stream::from_tcp(socket, bytes, 28e6).unwrap();
            let data: Vec<u8> = (0..258).map(|i| i as u8).collect();
            for part in data.chunks(3) {
                writer.write_all(part).unwrap();
            }
            drop(writer);
            assert_eq!(reader.read_raw(512).unwrap(), data);
            assert!(reader.read_raw(1).unwrap().is_empty());
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let writer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let socket = listener.accept().unwrap().0;
        let mut reader = crate::cxadc::Stream::from_tcp(socket, 1, 28e6).unwrap();
        drop(writer);
        assert!(reader.read_raw(1).unwrap().is_empty());
    }
    #[test]
    fn network_reader_stops_when_idle_and_rejects_truncated_wide_samples() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut writer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let socket = listener.accept().unwrap().0;
        let reader = crate::cxadc::Stream::from_tcp(socket, 1, 28e6).unwrap();
        let started = Instant::now();
        drop(reader);
        assert!(started.elapsed() < Duration::from_secs(1));
        let socket = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let mut peer = listener.accept().unwrap().0;
        let mut reader = crate::cxadc::Stream::from_tcp(socket, 2, 28e6).unwrap();
        peer.write_all(&[1, 2, 3]).unwrap();
        drop(peer);
        assert!(reader
            .read_raw(4)
            .unwrap_err()
            .contains("incomplete PCM sample"));
        // Keep the first connection alive until after the idle worker is joined.
        let _ = writer.write(&[0]);
    }
}
