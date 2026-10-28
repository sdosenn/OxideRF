//! Linux cxadc discovery and bounded, cancellable RF acquisition.
use std::{
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver, TrySendError},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const CHUNK: usize = 256 * 1024;
// Allocate payload blocks as they arrive, so memory is only used while the
// consumer is behind. At 28.636 MHz/u8 the default absorbs about 150 seconds
// of accumulated decoder lag while keeping memory use bounded.
pub const DEFAULT_BUFFER_MIB: usize = 4096;
pub const MIN_BUFFER_MIB: usize = 64;
/// Environment variable that sets the live RF queue limit in MiB.
pub const BUFFER_ENV: &str = "VHS_LIVE_BUFFER_MIB";

/// Live RF queue capacity in chunks, from `VHS_LIVE_BUFFER_MIB` (default 4096).
fn queue_chunks() -> Result<usize, String> {
    let mib = match std::env::var(BUFFER_ENV) {
        Ok(value) => value
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|&v| v >= MIN_BUFFER_MIB)
            .ok_or_else(|| {
                format!("VHS_LIVE_BUFFER_MIB must be an integer of at least {MIN_BUFFER_MIB}")
            })?,
        Err(_) => DEFAULT_BUFFER_MIB,
    };
    Ok(mib * 1024 * 1024 / CHUNK)
}

/// Installed physical memory in bytes, when the platform reports it.
pub fn installed_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        fs::read_to_string("/proc/meminfo")
            .ok()?
            .lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))?
            .trim()
            .trim_end_matches("kB")
            .trim()
            .parse::<u64>()
            .ok()
            .map(|kib| kib * 1024)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status.ullTotalPhys)
    }
    #[cfg(target_os = "macos")]
    {
        let mut bytes = 0u64;
        let mut size = std::mem::size_of::<u64>();
        let status = unsafe {
            libc::sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&mut bytes as *mut u64).cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        (status == 0).then_some(bytes)
    }
    #[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
    {
        None
    }
}

/// Whether cxadc capture exists on this platform (Linux cxadc, Windows cxadc-win).
pub const SUPPORTED: bool = cfg!(any(target_os = "linux", windows));
/// The first cxadc device path on this platform.
pub const DEFAULT_DEVICE: &str = if cfg!(windows) {
    r"\\.\cxadc0"
} else {
    "/dev/cxadc0"
};

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub path: PathBuf,
    pub tenbit: Option<bool>,
    pub frequency_mhz: Option<f64>,
    pub vmux: Option<u32>,
    pub level: Option<u32>,
}
impl DeviceInfo {
    pub fn sample_format(&self) -> Option<&'static str> {
        self.tenbit.map(|wide| if wide { "u16" } else { "u8" })
    }
    pub fn summary(&self) -> String {
        let rate = self
            .frequency_mhz
            .map(|v| format!("{v:.6} MHz"))
            .unwrap_or_else(|| "rate unknown".into());
        format!(
            "{} · {rate} · input {} · gain {}",
            self.sample_format().unwrap_or("format unknown"),
            self.vmux
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".into()),
            self.level
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".into())
        )
    }
}
/// `cxadcN` for `/dev/cxadcN` (Linux cxadc) or `\\.\cxadcN` (Windows
/// cxadc-win, github.com/JuniorIsAJitterbug/cxadc-win).
fn device_name(path: &Path) -> Option<&str> {
    let text = path.to_str()?;
    let name = match text.strip_prefix(r"\\.\") {
        Some(name) => name,
        None => path.file_name()?.to_str()?,
    };
    let suffix = name.strip_prefix("cxadc")?;
    (!suffix.is_empty() && suffix.bytes().all(|c| c.is_ascii_digit())).then_some(name)
}
#[cfg(windows)]
pub fn inspect(path: &Path) -> DeviceInfo {
    let config = windows_config(path);
    DeviceInfo {
        path: path.into(),
        tenbit: config.map(|c| c.tenbit != 0),
        // cxadc-win does not know the crystal or clock generator rate.
        frequency_mhz: None,
        vmux: config.map(|c| c.vmux),
        level: config.map(|c| c.level),
    }
}
/// `DEVICE_CONFIG` of cxadc-win (src/shared/cx_config.h).
#[cfg(windows)]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct WindowsConfig {
    vmux: u32,
    level: u32,
    tenbit: u8,
    sixdb: u8,
    center_offset: u32,
}
#[cfg(windows)]
fn windows_config(path: &Path) -> Option<WindowsConfig> {
    use std::os::windows::io::AsRawHandle;
    // CX_IOCTL_CONFIG_GET: CTL_CODE(FILE_DEVICE_UNKNOWN, 0x700, METHOD_BUFFERED,
    // FILE_READ_DATA) in cxadc-win src/shared/cx_ctl_codes.h.
    const CONFIG_GET: u32 = (0x22 << 16) | (1 << 14) | (0x700 << 2);
    let file = File::open(path).ok()?;
    let mut config = WindowsConfig::default();
    let mut returned = 0u32;
    let size = std::mem::size_of::<WindowsConfig>() as u32;
    // The handle is valid while `file` lives; the output buffer is `config`.
    let ok = unsafe {
        windows_sys::Win32::System::IO::DeviceIoControl(
            file.as_raw_handle() as _,
            CONFIG_GET,
            std::ptr::null(),
            0,
            (&mut config as *mut WindowsConfig).cast(),
            size,
            &mut returned,
            std::ptr::null_mut(),
        )
    };
    (ok != 0 && returned >= size).then_some(config)
}
#[cfg(not(windows))]
pub fn inspect(path: &Path) -> DeviceInfo {
    let root = device_name(path).map(|name| {
        PathBuf::from("/sys/class/cxadc")
            .join(name)
            .join("device/parameters")
    });
    inspect_parameters(path, root.as_deref())
}
#[cfg_attr(windows, allow(dead_code))]
fn inspect_parameters(path: &Path, root: Option<&Path>) -> DeviceInfo {
    let read = |key: &str| -> Option<u32> {
        fs::read_to_string(root?.join(key))
            .ok()?
            .trim()
            .parse()
            .ok()
    };
    let tenbit = read("tenbit").and_then(|v| match v {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    });
    let frequency_mhz = match (read("crystal"), read("tenxfsc"), tenbit) {
        (Some(crystal), Some(mode), Some(wide)) => {
            nominal_rate(crystal, mode, wide).map(|v| v / 1e6)
        }
        _ => None,
    };
    DeviceInfo {
        path: path.into(),
        tenbit,
        frequency_mhz,
        vmux: read("vmux"),
        level: read("level"),
    }
}
#[cfg_attr(windows, allow(dead_code))]
fn nominal_rate(crystal: u32, mode: u32, wide: bool) -> Option<f64> {
    let crystal = if crystal == 28_636_363 {
        315e6 / 11.0
    } else {
        crystal as f64
    };
    let rate = match mode {
        0 => crystal,
        1 => crystal * 1.25,
        2 => crystal * 40e6 / (315e6 / 11.0),
        10..=99 => mode as f64 * 1e6,
        100.. => mode as f64,
        _ => return None,
    };
    Some(rate / if wide { 2.0 } else { 1.0 })
}
#[cfg(windows)]
pub fn devices() -> Vec<DeviceInfo> {
    (0..8)
        .map(|n| PathBuf::from(format!(r"\\.\cxadc{n}")))
        .filter(|path| File::open(path).is_ok())
        .map(|path| inspect(&path))
        .collect()
}
#[cfg(not(windows))]
pub fn devices() -> Vec<DeviceInfo> {
    let mut devices: Vec<_> = fs::read_dir("/dev")
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| device_name(path).is_some())
        .filter(|path| is_char_device(path))
        .map(|path| inspect(&path))
        .collect();
    devices.sort_by_key(|d| d.path.clone());
    devices
}
/// A cxadc device (Linux, or cxadc-win on Windows) or an RF named pipe (Unix).
#[cfg(unix)]
pub fn is_stream(path: &Path) -> bool {
    use std::os::unix::fs::FileTypeExt;
    fs::metadata(path).is_ok_and(|m| m.file_type().is_char_device() || m.file_type().is_fifo())
}
#[cfg(windows)]
pub fn is_stream(path: &Path) -> bool {
    is_char_device(path)
}
#[cfg(not(any(unix, windows)))]
pub fn is_stream(_: &Path) -> bool {
    false
}
#[cfg(unix)]
fn is_char_device(path: &Path) -> bool {
    use std::os::unix::fs::FileTypeExt;
    fs::metadata(path).is_ok_and(|m| m.file_type().is_char_device())
}
#[cfg(windows)]
fn is_char_device(path: &Path) -> bool {
    path.to_str().is_some_and(|p| p.starts_with(r"\\.\")) && device_name(path).is_some()
}
#[cfg(not(any(unix, windows)))]
fn is_char_device(_: &Path) -> bool {
    false
}

#[derive(Default)]
struct State {
    stop: AtomicBool,
    bytes: AtomicU64,
    queued: AtomicUsize,
    error: Mutex<Option<String>>,
    timing: Mutex<std::collections::VecDeque<(Instant, u64)>>,
}
/// Only the Linux audio sender maps capture time to RF positions.
#[derive(Clone)]
pub(crate) struct Clock {
    state: Arc<State>,
    rate: f64,
    bytes: usize,
    epoch: Instant,
}
impl Clock {
    /// Fit acquisition sample positions, not TCP send times, to the host clock.
    pub(crate) fn position(&self, at: Instant) -> f64 {
        let points = self.state.timing.lock().unwrap();
        let Some(&(origin, _)) = points.front() else {
            return signed_seconds(at, self.epoch) * self.rate;
        };
        let n = points.len() as f64;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        for &(t, bytes) in points.iter() {
            let x = signed_seconds(t, origin);
            let y = bytes as f64 / self.bytes as f64;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let variance = n * sxx - sx * sx;
        let measured = if variance > 0.01 {
            (n * sxy - sx * sy) / variance
        } else {
            self.rate
        };
        let slope = if (self.rate * 0.98..=self.rate * 1.02).contains(&measured) {
            measured
        } else {
            self.rate
        };
        (sy - slope * sx) / n + slope * signed_seconds(at, origin)
    }
}
fn signed_seconds(at: Instant, origin: Instant) -> f64 {
    if at >= origin {
        (at - origin).as_secs_f64()
    } else {
        -(origin - at).as_secs_f64()
    }
}
impl State {
    fn enqueue(&self, sender: &mpsc::SyncSender<Vec<u8>>, chunk: Vec<u8>) -> bool {
        self.queued.fetch_add(chunk.len(), Ordering::AcqRel);
        match sender.try_send(chunk) {
            Ok(()) => true,
            Err(TrySendError::Full(chunk)) => {
                self.queued.fetch_sub(chunk.len(), Ordering::AcqRel);
                *self.error.lock().unwrap()=Some("Live RF buffer overflow: decoding or output storage cannot keep up. Capture stopped instead of continuing across a known sample gap. Raise the RF buffer (Settings, or VHS_LIVE_BUFFER_MIB) only to absorb longer stalls; sustained lag needs faster decoding or output.".into());
                false
            }
            Err(TrySendError::Disconnected(_)) => false,
        }
    }
}
fn capture_loop(
    mut file: impl Read,
    temporary_zero: bool,
    wait_for_first: bool,
    state: Arc<State>,
    sender: mpsc::SyncSender<Vec<u8>>,
) {
    let mut chunk = vec![0u8; CHUNK];
    let mut filled = 0;
    let mut seen_data = !wait_for_first;
    while !state.stop.load(Ordering::Acquire) {
        match file.read(&mut chunk[filled..]) {
            Ok(0) if temporary_zero || !seen_data => thread::sleep(Duration::from_millis(2)),
            Ok(0) => {
                if filled > 0 {
                    chunk.truncate(filled);
                    state.enqueue(&sender, chunk);
                }
                break;
            }
            Ok(n) => {
                seen_data = true;
                filled += n;
                state.bytes.fetch_add(n as u64, Ordering::Relaxed);
                if filled == CHUNK {
                    let mut timing = state.timing.lock().unwrap();
                    timing.push_back((Instant::now(), state.bytes.load(Ordering::Relaxed)));
                    if timing.len() > 128 {
                        timing.pop_front();
                    }
                    drop(timing);
                    // Aggregate short driver reads so the queue is bounded by bytes,
                    // rather than overflowing early on many tiny reads.
                    let complete = std::mem::replace(&mut chunk, vec![0u8; CHUNK]);
                    if !state.enqueue(&sender, complete) {
                        break;
                    }
                    filled = 0;
                }
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(2))
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => {
                *state.error.lock().unwrap() = Some(format!("Live RF read failed: {e}"));
                break;
            }
        }
    }
}
#[cfg(unix)]
fn open_nonblocking(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
}
/// cxadc-win reads block until DMA data is available, which the reader
/// thread tolerates; non-blocking mode would need a driver IOCTL.
#[cfg(windows)]
fn open_nonblocking(path: &Path) -> io::Result<File> {
    File::open(path)
}
#[cfg(not(any(unix, windows)))]
fn open_nonblocking(_: &Path) -> io::Result<File> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "cxadc capture needs Linux or Windows",
    ))
}
pub(crate) struct Stream {
    receiver: Receiver<Vec<u8>>,
    state: Arc<State>,
    worker: Option<JoinHandle<()>>,
    pending: Vec<u8>,
    offset: usize,
    bytes_per_sample: usize,
    started: Instant,
    reported: Instant,
    rate: f64,
    capacity: usize,
}
impl Stream {
    /// Received bytes, queued bytes and queue capacity in bytes.
    pub(crate) fn levels(&self) -> (u64, usize, usize) {
        (
            self.state.bytes.load(Ordering::Relaxed),
            self.state.queued.load(Ordering::Acquire),
            self.capacity,
        )
    }
    pub(crate) fn bytes_per_sample(&self) -> usize {
        self.bytes_per_sample
    }
    pub(crate) fn clock(&self) -> Clock {
        Clock {
            state: self.state.clone(),
            rate: self.rate,
            bytes: self.bytes_per_sample,
            epoch: self.started,
        }
    }
    pub(crate) fn open(path: &Path, format: &str, rate: f64) -> Result<Self, String> {
        if !is_stream(path) {
            return Err("Live input must be a cxadc character device or RF named pipe".into());
        }
        let bytes_per_sample = match format {
            "u8" => 1,
            "u16" => 2,
            _ => return Err("cxadc live input supports u8 or u16 LE samples".into()),
        };
        let file = open_nonblocking(path).map_err(|e| {
            format!(
                "Cannot open {}: {e}. Check device permissions and whether another capture is running.",
                path.display()
            )
        })?;
        let temporary_zero = is_char_device(path);
        Self::from_reader(
            file,
            temporary_zero,
            true,
            bytes_per_sample,
            rate,
            queue_chunks()?,
        )
    }
    pub(crate) fn from_tcp(
        socket: std::net::TcpStream,
        bytes_per_sample: usize,
        rate: f64,
    ) -> Result<Self, String> {
        socket.set_nonblocking(true).map_err(|e| e.to_string())?;
        Self::from_reader(
            socket,
            false,
            false,
            bytes_per_sample,
            rate,
            queue_chunks()?,
        )
    }
    fn from_reader(
        file: impl Read + Send + 'static,
        temporary_zero: bool,
        wait_for_first: bool,
        bytes_per_sample: usize,
        rate: f64,
        chunks: usize,
    ) -> Result<Self, String> {
        let (sender, receiver) = mpsc::sync_channel(chunks);
        println!(
            "LIVE_RF_BUFFER capacity_mib={} reserve_s={:.2}",
            chunks * CHUNK / (1024 * 1024),
            (chunks * CHUNK) as f64 / bytes_per_sample as f64 / rate
        );
        let state = Arc::new(State::default());
        let worker_state = state.clone();
        let worker = thread::Builder::new()
            .name("cxadc-reader".into())
            .spawn(move || capture_loop(file, temporary_zero, wait_for_first, worker_state, sender))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            receiver,
            state,
            worker: Some(worker),
            pending: Vec::new(),
            offset: 0,
            bytes_per_sample,
            started: Instant::now(),
            reported: Instant::now(),
            rate,
            capacity: chunks * CHUNK,
        })
    }
    pub(crate) fn read(&mut self, samples: usize) -> Result<Vec<f32>, String> {
        let bytes = self.read_raw(samples)?;
        Ok(if self.bytes_per_sample == 1 {
            bytes
                .into_iter()
                .map(|v| (v as f32 - 128.0) / 128.0)
                .collect()
        } else {
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| (u16::from_le_bytes(*b) as f32 - 32768.0) / 32768.0)
                .collect()
        })
    }
    pub(crate) fn read_raw(&mut self, samples: usize) -> Result<Vec<u8>, String> {
        let wanted = samples * self.bytes_per_sample;
        let mut bytes = Vec::with_capacity(wanted);
        while bytes.len() < wanted && !crate::native::stop_requested() {
            if let Some(error) = self.state.error.lock().map_err(|e| e.to_string())?.clone() {
                return Err(error);
            }
            if self.offset < self.pending.len() {
                let n = (wanted - bytes.len()).min(self.pending.len() - self.offset);
                bytes.extend_from_slice(&self.pending[self.offset..self.offset + n]);
                self.offset += n;
                self.state.queued.fetch_sub(n, Ordering::AcqRel);
            } else {
                match self.receiver.recv_timeout(Duration::from_millis(50)) {
                    Ok(chunk) => {
                        self.pending = chunk;
                        self.offset = 0;
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        if let Some(error) = self.state.error.lock().unwrap().clone() {
                            return Err(error);
                        }
                        break;
                    }
                }
            }
        }
        if !bytes.len().is_multiple_of(self.bytes_per_sample) {
            if crate::native::stop_requested() {
                bytes.truncate(bytes.len() / self.bytes_per_sample * self.bytes_per_sample);
            } else {
                return Err("Live RF ended with an incomplete PCM sample".into());
            }
        }
        if self.reported.elapsed() >= Duration::from_secs(1) {
            let total = self.state.bytes.load(Ordering::Relaxed);
            let queued = self.state.queued.load(Ordering::Acquire);
            println!(
                "LIVE_RF received_mib={:.1} rate_mib_s={:.2} queue_ms={:.1}",
                total as f64 / 1048576.0,
                total as f64 / 1048576.0 / self.started.elapsed().as_secs_f64(),
                queued as f64 / self.bytes_per_sample as f64 / self.rate * 1000.0
            );
            self.reported = Instant::now();
        }
        Ok(bytes)
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        self.state.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn audio_clock_uses_acquisition_positions_and_tracks_rf_clock_drift() {
        let state = Arc::new(State::default());
        let epoch = Instant::now();
        let rate = 315e6 / 11.0;
        let actual = rate * 1.0002;
        let initial = 3u64 << 32;
        for i in 1..=128 {
            state.timing.lock().unwrap().push_back((
                epoch + Duration::from_millis(i * 10),
                initial + (actual * i as f64 * 0.01).round() as u64,
            ));
        }
        let clock = Clock {
            state,
            rate,
            bytes: 1,
            epoch,
        };
        let position = clock.position(epoch + Duration::from_millis(650));
        assert!((position - (initial as f64 + actual * 0.65)).abs() < 1.0);
    }
    #[test]
    fn cxadc_clock_and_format_follow_tenbit_decimation() {
        assert_eq!(nominal_rate(28_636_363, 0, false), Some(315e6 / 11.0));
        assert_eq!(nominal_rate(28_636_363, 0, true), Some(315e6 / 22.0));
        assert_eq!(nominal_rate(40_000_000, 20, true), Some(10e6));
        assert_eq!(nominal_rate(40_000_000, 40_000_000, false), Some(40e6));
        assert_eq!(nominal_rate(40_000_000, 5, false), None);
        assert!(device_name(Path::new("/dev/cxadc2")).is_some());
        assert_eq!(device_name(Path::new(r"\\.\cxadc1")), Some("cxadc1"));
        assert!(device_name(Path::new("/dev/cxadc../../bad")).is_none());
    }
}

#[cfg(all(test, unix))]
mod stream_tests {
    use super::*;
    use std::{
        io::Write,
        os::{
            fd::{FromRawFd, IntoRawFd},
            unix::net::UnixStream,
        },
    };

    // Small enough for overflow tests, larger than the stall test below.
    const TEST_QUEUE: usize = 128;

    fn pair(bytes: usize) -> (Stream, UnixStream) {
        let (reader, writer) = UnixStream::pair().unwrap();
        reader.set_nonblocking(true).unwrap();
        // Ownership of the socket descriptor moves into File exactly once.
        let file = unsafe { File::from_raw_fd(reader.into_raw_fd()) };
        (
            Stream::from_reader(file, false, true, bytes, 315e6 / 11.0, TEST_QUEUE).unwrap(),
            writer,
        )
    }
    #[test]
    fn split_u16_samples_survive_short_reads_and_eof() {
        let (mut stream, mut writer) = pair(2);
        let producer = thread::spawn(move || {
            writer.write_all(&[0, 0, 0]).unwrap();
            thread::sleep(Duration::from_millis(10));
            writer.write_all(&[128, 255, 255]).unwrap();
        });
        assert_eq!(stream.read(3).unwrap(), vec![-1.0, 0.0, 32767.0 / 32768.0]);
        assert!(stream.read(1).unwrap().is_empty());
        producer.join().unwrap();
    }
    #[test]
    fn truncated_live_sample_is_an_error() {
        let (mut stream, mut writer) = pair(2);
        writer.write_all(&[0, 128, 255]).unwrap();
        drop(writer);
        assert!(stream
            .read(2)
            .unwrap_err()
            .contains("incomplete PCM sample"));
    }
    #[test]
    fn temporary_stall_beyond_old_queue_capacity_preserves_all_samples() {
        let (mut stream, mut writer) = pair(1);
        let chunks = 64;
        let producer = thread::spawn(move || {
            for index in 0..chunks {
                writer.write_all(&vec![index as u8; CHUNK]).unwrap();
            }
        });
        let started = Instant::now();
        while stream.state.queued.load(Ordering::Acquire) < (chunks - 1) * CHUNK {
            assert!(stream.state.error.lock().unwrap().is_none());
            assert!(started.elapsed() < Duration::from_secs(5));
            thread::sleep(Duration::from_millis(2));
        }
        producer.join().unwrap();
        for index in 0..chunks {
            let bytes = stream.read_raw(CHUNK).unwrap();
            assert_eq!(bytes.len(), CHUNK);
            assert!(bytes.iter().all(|value| *value == index as u8));
        }
        assert!(stream.read_raw(1).unwrap().is_empty());
    }
    #[test]
    fn stalled_consumer_reports_overflow_instead_of_silent_sample_loss() {
        let (mut stream, mut writer) = pair(1);
        let producer = thread::spawn(move || {
            let chunk = vec![128u8; CHUNK];
            for _ in 0..TEST_QUEUE + 4 {
                if writer.write_all(&chunk).is_err() {
                    break;
                }
            }
        });
        let started = Instant::now();
        while stream.state.error.lock().unwrap().is_none() {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "overflow was not detected"
            );
            thread::sleep(Duration::from_millis(2));
        }
        assert!(stream.read(1).unwrap_err().contains("buffer overflow"));
        drop(stream);
        producer.join().unwrap();
    }
    #[test]
    fn idle_capture_worker_can_be_joined_without_incoming_samples() {
        let (stream, _writer) = pair(1);
        drop(stream);
    }
}
