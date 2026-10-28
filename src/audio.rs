//! Native stereo PCM capture, monitoring and timestamped RF/audio sidecars.
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    fs::File,
    io::{Read, Write},
    net::{Shutdown, TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
#[cfg(target_os = "linux")]
use std::{
    ffi::{CStr, CString},
    sync::OnceLock,
};
pub const RATE: u32 = 48_000;
const FRAMES: usize = 960;
#[cfg(target_os = "linux")]
const BYTES: usize = FRAMES * 4;
const MAGIC: &[u8; 8] = b"VHSAUD01";

#[derive(Clone)]
pub struct Device {
    pub name: String,
    pub description: String,
}
pub fn devices() -> Vec<Device> {
    let mut result = vec![Device {
        name: "default".into(),
        description: "System default input".into(),
    }];
    #[cfg(target_os = "linux")]
    if let Ok(output) = Command::new("pactl")
        .args(["--format=json", "list", "sources"])
        .output()
    {
        if let Ok(Value::Array(items)) = serde_json::from_slice(&output.stdout) {
            for item in items {
                if let Some(name) = item["name"].as_str() {
                    result.push(Device {
                        name: name.into(),
                        description: item["description"].as_str().unwrap_or(name).into(),
                    });
                }
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        use cpal::traits::HostTrait;
        if let Ok(inputs) = cpal::default_host().input_devices() {
            result.extend(inputs.map(|device| {
                let name = device.to_string();
                Device {
                    description: name.clone(),
                    name,
                }
            }));
        }
    }
    result
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct Spec {
    format: i32,
    rate: u32,
    channels: u8,
}
#[cfg(target_os = "linux")]
#[repr(C)]
struct Attr {
    maxlength: u32,
    tlength: u32,
    prebuf: u32,
    minreq: u32,
    fragsize: u32,
}
#[cfg(target_os = "linux")]
type New = unsafe extern "C" fn(
    *const i8,
    *const i8,
    i32,
    *const i8,
    *const i8,
    *const Spec,
    *const libc::c_void,
    *const Attr,
    *mut i32,
) -> *mut libc::c_void;
#[cfg(target_os = "linux")]
type Io = unsafe extern "C" fn(*mut libc::c_void, *mut libc::c_void, usize, *mut i32) -> i32;
#[cfg(target_os = "linux")]
struct Api {
    new: New,
    read: Io,
    write: Io,
    free: unsafe extern "C" fn(*mut libc::c_void),
    latency: unsafe extern "C" fn(*mut libc::c_void, *mut i32) -> u64,
    error: unsafe extern "C" fn(i32) -> *const i8,
}
#[cfg(target_os = "linux")]
fn api() -> Result<&'static Api, String> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        // The library stays loaded for the process lifetime; each stream has one owner.
        unsafe {
            let handle=libc::dlopen(c"libpulse-simple.so.0".as_ptr(),libc::RTLD_NOW);
            if handle.is_null(){return Err("Audio requires libpulse-simple and a running PulseAudio/PipeWire-Pulse service".into());}
            macro_rules! symbol {($name:literal,$ty:ty)=>{{let p=libc::dlsym(handle,concat!($name,"\0").as_ptr().cast());if p.is_null(){return Err(format!("Missing PulseAudio symbol {}",$name));} std::mem::transmute::<*mut libc::c_void,$ty>(p)}};}
            Ok(Api{new:symbol!("pa_simple_new",New),read:symbol!("pa_simple_read",Io),write:symbol!("pa_simple_write",Io),free:symbol!("pa_simple_free",unsafe extern "C" fn(*mut libc::c_void)),latency:symbol!("pa_simple_get_latency",unsafe extern "C" fn(*mut libc::c_void,*mut i32)->u64),error:symbol!("pa_strerror",unsafe extern "C" fn(i32)->*const i8)})
        }
    }).as_ref().map_err(Clone::clone)
}
#[cfg(target_os = "linux")]
struct Pulse {
    handle: *mut libc::c_void,
    api: &'static Api,
}
#[cfg(target_os = "linux")]
// Ownership transfers into a worker; libpulse-simple does not share this handle.
unsafe impl Send for Pulse {}
#[cfg(target_os = "linux")]
impl Pulse {
    fn open(source: Option<&str>, record: bool) -> Result<Self, String> {
        let api = api()?;
        let source = source
            .filter(|s| *s != "default")
            .map(CString::new)
            .transpose()
            .map_err(|e| e.to_string())?;
        let spec = Spec {
            format: 3,
            rate: RATE,
            channels: 2,
        };
        let attr = Attr {
            maxlength: u32::MAX,
            tlength: (BYTES * 4) as u32,
            prebuf: u32::MAX,
            minreq: BYTES as u32,
            fragsize: BYTES as u32,
        };
        let mut error = 0;
        let handle = unsafe {
            (api.new)(
                std::ptr::null(),
                c"Oxide RF".as_ptr(),
                if record { 2 } else { 1 },
                source.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                c"VCR audio".as_ptr(),
                &spec,
                std::ptr::null(),
                &attr,
                &mut error,
            )
        };
        if handle.is_null() {
            return Err(format!(
                "Audio device: {}",
                unsafe { CStr::from_ptr((api.error)(error)) }.to_string_lossy()
            ));
        }
        Ok(Self { handle, api })
    }
    fn io(&self, bytes: &mut [u8], record: bool) -> Result<(), String> {
        let mut error = 0;
        let status = unsafe {
            (if record {
                self.api.read
            } else {
                self.api.write
            })(
                self.handle,
                bytes.as_mut_ptr().cast(),
                bytes.len(),
                &mut error,
            )
        };
        if status < 0 {
            Err(format!(
                "Audio I/O: {}",
                unsafe { CStr::from_ptr((self.api.error)(error)) }.to_string_lossy()
            ))
        } else {
            Ok(())
        }
    }
    fn latency(&self) -> Result<Duration, String> {
        let mut error = 0;
        let latency = unsafe { (self.api.latency)(self.handle, &mut error) };
        if latency == u64::MAX {
            Err("Cannot obtain audio capture latency".into())
        } else {
            Ok(Duration::from_micros(latency))
        }
    }
}
#[cfg(target_os = "linux")]
impl Drop for Pulse {
    fn drop(&mut self) {
        unsafe {
            (self.api.free)(self.handle);
        }
    }
}

fn read_cancel(
    socket: &mut TcpStream,
    bytes: &mut [u8],
    stop: &AtomicBool,
) -> Result<bool, String> {
    let mut offset = 0;
    while offset < bytes.len() {
        if stop.load(Ordering::Relaxed) {
            return Ok(false);
        }
        match socket.read(&mut bytes[offset..]) {
            Ok(0) if stop.load(Ordering::Relaxed) => return Ok(false),
            Ok(0) if offset == 0 => return Ok(false),
            Ok(0) => return Err("Truncated audio packet".into()),
            Ok(n) => offset += n,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::Interrupted
                ) => {}
            Err(_) if stop.load(Ordering::Relaxed) => return Ok(false),
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(true)
}
/// Accepts the receiver's audio connection and checks its session token.
fn accept_receiver(listener: &TcpListener, token: u64) -> Result<TcpStream, String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut socket = loop {
        if crate::native::stop_requested() {
            return Err("Audio connection cancelled".into());
        }
        if Instant::now() > deadline {
            return Err("Audio receiver did not connect".into());
        }
        match listener.accept() {
            Ok((socket, _)) => break socket,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(10))
            }
            Err(e) => return Err(e.to_string()),
        }
    };
    // Windows sockets inherit the listener's non-blocking mode.
    socket.set_nonblocking(false).map_err(|e| e.to_string())?;
    socket
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| e.to_string())?;
    socket
        .set_write_timeout(Some(Duration::from_millis(500)))
        .map_err(|e| e.to_string())?;
    let mut hello = [0u8; 16];
    socket.read_exact(&mut hello).map_err(|e| e.to_string())?;
    if &hello[..8] != MAGIC || u64::from_le_bytes(hello[8..].try_into().unwrap()) != token {
        return Err("Audio session mismatch".into());
    }
    Ok(socket)
}
/// Block header: RF position of the first frame, frame index and frame count.
/// A frame count of zero marks the end of the session.
fn block_header(rf: f64, position: u64, frames: u32) -> [u8; 20] {
    let mut header = [0u8; 20];
    header[..8].copy_from_slice(&rf.to_le_bytes());
    header[8..16].copy_from_slice(&position.to_le_bytes());
    header[16..].copy_from_slice(&frames.to_le_bytes());
    header
}

#[cfg(target_os = "linux")]
pub(crate) struct SenderReady {
    socket: TcpStream,
    pulse: Pulse,
}
#[cfg(target_os = "linux")]
impl SenderReady {
    pub(crate) fn accept(listener: &TcpListener, token: u64, source: &str) -> Result<Self, String> {
        let mut socket = accept_receiver(listener, token)?;
        let pulse = Pulse::open(Some(source), true)?;
        socket.write_all(MAGIC).map_err(|e| e.to_string())?;
        Ok(Self { socket, pulse })
    }
    pub(crate) fn start(self, clock: crate::cxadc::Clock) -> Result<Sender, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let worker = thread::Builder::new()
            .name("audio-send".into())
            .spawn(move || {
                let mut socket = self.socket;
                let pulse = self.pulse;
                let mut pcm = vec![0; BYTES];
                let mut position = 0u64;
                while !worker_stop.load(Ordering::Relaxed) {
                    pulse.io(&mut pcm, true)?;
                    let at = Instant::now()
                        .checked_sub(
                            pulse.latency()? + Duration::from_secs_f64(FRAMES as f64 / RATE as f64),
                        )
                        .ok_or("Invalid audio latency")?;
                    socket
                        .write_all(&block_header(clock.position(at), position, FRAMES as u32))
                        .and_then(|()| socket.write_all(&pcm))
                        .map_err(|e| format!("Audio send: {e}"))?;
                    position += FRAMES as u64;
                }
                socket
                    .write_all(&block_header(clock.position(Instant::now()), position, 0))
                    .map_err(|e| e.to_string())?;
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        Ok(Sender {
            stop,
            worker: Some(worker),
        })
    }
}

/// Sends VCR audio captured from a cpal input device (WASAPI on Windows,
/// CoreAudio on macOS). Each callback's capture timestamp anchors the RF
/// position of the blocks it produces.
#[cfg(not(target_os = "linux"))]
pub(crate) struct SenderReady {
    socket: TcpStream,
    source: String,
}
#[cfg(not(target_os = "linux"))]
impl SenderReady {
    pub(crate) fn accept(listener: &TcpListener, token: u64, source: &str) -> Result<Self, String> {
        let mut socket = accept_receiver(listener, token)?;
        input_device(source)?;
        socket.write_all(MAGIC).map_err(|e| e.to_string())?;
        Ok(Self {
            socket,
            source: source.into(),
        })
    }
    pub(crate) fn start(self, clock: crate::cxadc::Clock) -> Result<Sender, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let (opened, open_result) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("audio-send".into())
            .spawn(move || {
                let mut socket = self.socket;
                // The cpal stream is not Send, so it lives on this thread.
                let capture = Capture::open(&self.source);
                opened
                    .send(capture.as_ref().map(|_| ()).map_err(Clone::clone))
                    .ok();
                let capture = capture?;
                let mut position = 0u64;
                capture.run(&worker_stop, |at, pcm| {
                    socket
                        .write_all(&block_header(clock.position(at), position, FRAMES as u32))
                        .and_then(|()| socket.write_all(pcm))
                        .map_err(|e| format!("Audio send: {e}"))?;
                    position += FRAMES as u64;
                    Ok(())
                })?;
                drop(capture);
                socket
                    .write_all(&block_header(clock.position(Instant::now()), position, 0))
                    .map_err(|e| e.to_string())?;
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        if let Err(e) = open_result
            .recv()
            .unwrap_or_else(|_| Err("Audio input thread failed".into()))
        {
            worker.join().ok();
            return Err(e);
        }
        Ok(Sender {
            stop,
            worker: Some(worker),
        })
    }
}

pub(crate) struct Sender {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<(), String>>>,
}
impl Drop for Sender {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            if let Ok(Err(e)) = worker.join() {
                eprintln!("AUDIO_SERVER session_ended: {e}");
            }
        }
    }
}

/// Sample formats cpal streams are built with, best first.
fn sample_rank(format: cpal::SampleFormat) -> Option<u8> {
    match format {
        cpal::SampleFormat::F32 => Some(0),
        cpal::SampleFormat::I16 => Some(1),
        cpal::SampleFormat::I32 => Some(2),
        _ => None,
    }
}
/// The best 48 kHz stereo-or-more configuration in a supported sample format.
fn preferred_config(
    configs: impl Iterator<Item = cpal::SupportedStreamConfigRange>,
) -> Option<cpal::SupportedStreamConfig> {
    configs
        .filter(|c| c.channels() >= 2 && sample_rank(c.sample_format()).is_some())
        .filter_map(|c| c.try_with_sample_rate(RATE))
        .min_by_key(|c| sample_rank(c.sample_format()))
}

/// The cpal input device `devices` lists under `name`; "default" is the
/// system default input.
#[cfg_attr(target_os = "linux", allow(dead_code))]
fn input_device(name: &str) -> Result<cpal::Device, String> {
    use cpal::traits::HostTrait;
    let host = cpal::default_host();
    if name == "default" {
        return host
            .default_input_device()
            .ok_or_else(|| "No audio input device".into());
    }
    host.input_devices()
        .map_err(|e| e.to_string())?
        .find(|device| device.to_string() == name)
        .ok_or_else(|| format!("Audio input device not found: {name}"))
}

/// Interleaved samples from one input callback and the instant the first
/// of them was captured.
type Chunk = (Instant, Vec<f32>);

/// A running cpal input stream. Linux uses PulseAudio for the sender, so there
/// this only backs the capture test.
#[cfg_attr(target_os = "linux", allow(dead_code))]
struct Capture {
    _stream: cpal::Stream,
    chunks: mpsc::Receiver<Chunk>,
    channels: usize,
    rate: u32,
}
#[cfg_attr(target_os = "linux", allow(dead_code))]
impl Capture {
    fn open(name: &str) -> Result<Self, String> {
        use cpal::traits::{DeviceTrait, StreamTrait};
        let device = input_device(name)?;
        let supported = preferred_config(
            device
                .supported_input_configs()
                .map_err(|e| e.to_string())?,
        )
        .map_or_else(
            || device.default_input_config().map_err(|e| e.to_string()),
            Ok,
        )?;
        let config = supported.config();
        let (channels, rate) = (config.channels as usize, config.sample_rate);
        let (send, chunks) = mpsc::channel();
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => input_stream(&device, config, send, |s: f32| s),
            cpal::SampleFormat::I16 => {
                input_stream(&device, config, send, |s: i16| s as f32 / 32768.0)
            }
            cpal::SampleFormat::I32 => input_stream(&device, config, send, |s: i32| {
                (s as f64 / 2147483648.0) as f32
            }),
            format => return Err(format!("Unsupported audio input format {format}")),
        }?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Self {
            _stream: stream,
            chunks,
            channels,
            rate,
        })
    }
    /// Delivers 48 kHz stereo blocks until `stop` is set.
    fn run(
        &self,
        stop: &AtomicBool,
        mut block: impl FnMut(Instant, &[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut blocks = Blocks::new(self.channels, self.rate);
        let mut last = Instant::now();
        while !stop.load(Ordering::Relaxed) {
            match self.chunks.recv_timeout(Duration::from_millis(100)) {
                Ok((at, samples)) => {
                    last = Instant::now();
                    blocks.push(at, &samples, &mut block)?;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if last.elapsed() > Duration::from_secs(5) {
                        return Err("Audio input stopped delivering samples".into());
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("Audio input stream closed".into())
                }
            }
        }
        Ok(())
    }
}
#[cfg_attr(target_os = "linux", allow(dead_code))]
fn input_stream<T: cpal::SizedSample + 'static>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    send: mpsc::Sender<Chunk>,
    convert: fn(T) -> f32,
) -> Result<cpal::Stream, String> {
    use cpal::traits::DeviceTrait;
    device
        .build_input_stream::<T, _, _>(
            config,
            move |data: &[T], info: &cpal::InputCallbackInfo| {
                // The callback runs now; the data was captured `delay` earlier.
                let stamp = info.timestamp();
                let delay = stamp.callback.duration_since(stamp.capture);
                let now = Instant::now();
                let at = now.checked_sub(delay).unwrap_or(now);
                send.send((at, data.iter().map(|&s| convert(s)).collect()))
                    .ok();
            },
            |e| eprintln!("Audio input: {e}"),
            None,
        )
        .map_err(|e| e.to_string())
}

/// Converts timestamped interleaved chunks at the device rate into blocks of
/// FRAMES 48 kHz stereo 16-bit frames with the capture instant of their first
/// frame. Mono is duplicated, channels beyond two are dropped, and other rates
/// are interpolated linearly.
#[cfg_attr(target_os = "linux", allow(dead_code))]
struct Blocks {
    channels: usize,
    rate: f64,
    step: f64,
    /// Pending input frames; `input[0]` has absolute index `start`.
    input: Vec<[f32; 2]>,
    start: u64,
    /// Capture instant of an absolute input frame index.
    anchor: Option<(Instant, u64)>,
    /// Output frames produced so far.
    produced: u64,
    pcm: Vec<u8>,
    pcm_at: Option<Instant>,
}
#[cfg_attr(target_os = "linux", allow(dead_code))]
impl Blocks {
    fn new(channels: usize, rate: u32) -> Self {
        Self {
            channels: channels.max(1),
            rate: rate as f64,
            step: rate as f64 / RATE as f64,
            input: Vec::new(),
            start: 0,
            anchor: None,
            produced: 0,
            pcm: Vec::with_capacity(FRAMES * 4),
            pcm_at: None,
        }
    }
    fn push(
        &mut self,
        at: Instant,
        samples: &[f32],
        block: &mut impl FnMut(Instant, &[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        self.anchor = Some((at, self.start + self.input.len() as u64));
        self.input.extend(
            samples
                .chunks_exact(self.channels)
                .map(|frame| [frame[0], frame[if self.channels > 1 { 1 } else { 0 }]]),
        );
        let end = self.start + self.input.len() as u64;
        loop {
            // Computed from the output count so rounding cannot accumulate.
            let position = self.produced as f64 * self.step;
            let index = position as u64;
            let t = (position - index as f64) as f32;
            if index + u64::from(t > 0.0) >= end {
                break;
            }
            let a = self.input[(index - self.start) as usize];
            let b = if t > 0.0 {
                self.input[(index + 1 - self.start) as usize]
            } else {
                a
            };
            if self.pcm.is_empty() {
                let (instant, frame) = self.anchor.unwrap();
                let offset = (position - frame as f64) / self.rate;
                self.pcm_at = Some(if offset >= 0.0 {
                    instant + Duration::from_secs_f64(offset)
                } else {
                    instant - Duration::from_secs_f64(-offset)
                });
            }
            for channel in 0..2 {
                let value = a[channel] + (b[channel] - a[channel]) * t;
                let sample = (value.clamp(-1.0, 1.0) * 32767.0).round() as i16;
                self.pcm.extend_from_slice(&sample.to_le_bytes());
            }
            self.produced += 1;
            if self.pcm.len() == FRAMES * 4 {
                block(self.pcm_at.take().unwrap(), &self.pcm)?;
                self.pcm.clear();
            }
        }
        let keep = ((self.produced as f64 * self.step) as u64).saturating_sub(self.start);
        let drop = (keep as usize).min(self.input.len());
        self.input.drain(..drop);
        self.start += drop as u64;
        Ok(())
    }
}

/// Plays received 48 kHz stereo PCM on the default output device of any
/// platform, converted to the device's rate and channel count. Underruns play
/// silence; more than half a second of backlog drops the oldest samples.
struct Monitor {
    buffer: Arc<Mutex<VecDeque<f32>>>,
    _stream: cpal::Stream,
    channels: usize,
    step: f64,
    position: f64,
    last: [f32; 2],
    limit: usize,
}
impl Monitor {
    fn open() -> Result<Self, String> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        let device = cpal::default_host()
            .default_output_device()
            .ok_or("No audio output device")?;
        // Prefer 48 kHz stereo-or-more in a supported sample format; otherwise
        // use the device default and resample.
        let supported = preferred_config(
            device
                .supported_output_configs()
                .map_err(|e| e.to_string())?,
        )
        .map_or_else(
            || device.default_output_config().map_err(|e| e.to_string()),
            Ok,
        )?;
        let config = supported.config();
        let channels = config.channels as usize;
        let rate = config.sample_rate;
        let buffer = Arc::new(Mutex::new(VecDeque::new()));
        let source = buffer.clone();
        let fill = move |out: &mut [f32]| {
            let mut queue = source.lock().unwrap();
            for sample in out {
                *sample = queue.pop_front().unwrap_or(0.0);
            }
        };
        let error = |e| eprintln!("Audio monitor: {e}");
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => device.build_output_stream::<f32, _, _>(
                config,
                move |out: &mut [f32], _: &cpal::OutputCallbackInfo| fill(out),
                error,
                None,
            ),
            cpal::SampleFormat::I16 => {
                let mut scratch = Vec::new();
                device.build_output_stream::<i16, _, _>(
                    config,
                    move |out: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        scratch.resize(out.len(), 0.0);
                        fill(&mut scratch);
                        for (o, s) in out.iter_mut().zip(&scratch) {
                            *o = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                        }
                    },
                    error,
                    None,
                )
            }
            cpal::SampleFormat::I32 => {
                let mut scratch = Vec::new();
                device.build_output_stream::<i32, _, _>(
                    config,
                    move |out: &mut [i32], _: &cpal::OutputCallbackInfo| {
                        scratch.resize(out.len(), 0.0);
                        fill(&mut scratch);
                        for (o, s) in out.iter_mut().zip(&scratch) {
                            *o = (s.clamp(-1.0, 1.0) as f64 * i32::MAX as f64) as i32;
                        }
                    },
                    error,
                    None,
                )
            }
            format => return Err(format!("Unsupported audio output format {format}")),
        }
        .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Self {
            buffer,
            _stream: stream,
            channels,
            step: RATE as f64 / rate as f64,
            position: 0.0,
            last: [0.0; 2],
            limit: rate as usize * channels / 2,
        })
    }
    /// Queue little-endian 16-bit stereo PCM at 48 kHz.
    fn push(&mut self, pcm: &[u8]) {
        let frames: Vec<[f32; 2]> = std::iter::once(self.last)
            .chain(pcm.as_chunks::<4>().0.iter().map(|f| {
                [
                    i16::from_le_bytes([f[0], f[1]]) as f32 / 32768.0,
                    i16::from_le_bytes([f[2], f[3]]) as f32 / 32768.0,
                ]
            }))
            .collect();
        let mut out = Vec::new();
        // Linear interpolation from 48 kHz to the device rate.
        while self.position + 1.0 < frames.len() as f64 {
            let index = self.position as usize;
            let t = (self.position - index as f64) as f32;
            let [a, b] = [frames[index], frames[index + 1]];
            let frame = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            out.extend(frame);
            out.extend(std::iter::repeat_n(0.0, self.channels - 2));
            self.position += self.step;
        }
        self.position -= (frames.len() - 1) as f64;
        self.last = *frames.last().unwrap();
        let mut queue = self.buffer.lock().unwrap();
        queue.extend(out);
        let excess = queue.len().saturating_sub(self.limit);
        queue.drain(..excess - excess % self.channels);
    }
}

pub(crate) struct Receiver {
    stop: Arc<AtomicBool>,
    control: TcpStream,
    worker: Option<JoinHandle<Result<(), String>>>,
}
impl Receiver {
    pub(crate) fn connect(
        address: &str,
        port: u16,
        token: u64,
        base: Option<&Path>,
        monitor: bool,
        rate: f64,
    ) -> Result<Self, String> {
        use std::net::ToSocketAddrs;
        let host = address
            .rsplit_once(':')
            .ok_or("Invalid audio server address")?
            .0;
        let endpoint = format!("{host}:{port}");
        let addresses = endpoint.to_socket_addrs().map_err(|e| e.to_string())?;
        let mut socket = None;
        for address in addresses {
            if let Ok(stream) = TcpStream::connect_timeout(&address, Duration::from_secs(3)) {
                socket = Some(stream);
                break;
            }
        }
        let mut socket = socket.ok_or("Cannot connect to server audio port")?;
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .map_err(|e| e.to_string())?;
        let mut hello = [0u8; 16];
        hello[..8].copy_from_slice(MAGIC);
        hello[8..].copy_from_slice(&token.to_le_bytes());
        socket.write_all(&hello).map_err(|e| e.to_string())?;
        let mut ack = [0u8; 8];
        socket.read_exact(&mut ack).map_err(|e| e.to_string())?;
        if &ack != MAGIC {
            return Err("Unsupported audio protocol".into());
        }
        socket
            .set_read_timeout(Some(Duration::from_millis(100)))
            .map_err(|e| e.to_string())?;
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(16);
        // cpal streams are not Send on every platform: open and own it in one thread.
        let playback_worker = monitor.then(|| {
            thread::spawn(move || {
                match Monitor::open() {
                    Ok(mut output) => {
                        while let Ok(bytes) = rx.recv() {
                            output.push(&bytes);
                        }
                    }
                    Err(e) => {
                        eprintln!("Audio monitor unavailable, recording continues: {e}");
                        while rx.recv().is_ok() {}
                    }
                }
                Ok::<_, String>(())
            })
        });
        let base = base.map(Path::to_path_buf);
        let recording = base.is_some();
        let mut wav = if let Some(base) = &base {
            let path = PathBuf::from(format!("{}.audio.wav", base.display()));
            let timing = PathBuf::from(format!("{}.audio.json", base.display()));
            if timing.exists() {
                return Err(format!(
                    "Audio timing output already exists: {}",
                    timing.display()
                ));
            }
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|e| format!("Audio output {}: {e}", path.display()))?;
            Some(
                hound::WavWriter::new(
                    std::io::BufWriter::new(file),
                    hound::WavSpec {
                        channels: 2,
                        sample_rate: RATE,
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    },
                )
                .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let control = socket.try_clone().map_err(|e| e.to_string())?;
        let worker = thread::Builder::new()
            .name("audio-receive".into())
            .spawn(move || {
                let mut position = 0u64;
                let mut points = Vec::new();
                let mut monitor_drops = 0;
                let result = (|| -> Result<(), String> {
                    loop {
                        let mut header = [0u8; 20];
                        if !read_cancel(&mut socket, &mut header, &worker_stop)? {
                            if !worker_stop.load(Ordering::Relaxed) {
                                return Err("Audio stream closed without an end marker".into());
                            }
                            break;
                        }
                        let rf = f64::from_le_bytes(header[..8].try_into().unwrap());
                        let index = u64::from_le_bytes(header[8..16].try_into().unwrap());
                        let frames = u32::from_le_bytes(header[16..].try_into().unwrap()) as usize;
                        if !rf.is_finite() || index != position || frames > RATE as usize {
                            return Err("Invalid or discontinuous audio packet".into());
                        }
                        if frames == 0 {
                            break;
                        }
                        let mut pcm = vec![0; frames * 4];
                        if !read_cancel(&mut socket, &mut pcm, &worker_stop)? {
                            if !worker_stop.load(Ordering::Relaxed) {
                                return Err("Truncated audio payload".into());
                            }
                            break;
                        }
                        if let Some(writer) = &mut wav {
                            for b in pcm.as_chunks::<2>().0 {
                                writer.write_sample(i16::from_le_bytes(*b)).map_err(|e| e.to_string())?;
                            }
                            points.push(json!([index, rf]));
                        }
                        if monitor && tx.try_send(pcm).is_err() {
                            monitor_drops += 1;
                        }
                        position += frames as u64;
                    }
                    Ok(())
                })();
                if let Some(writer) = wav {
                    writer.finalize().map_err(|e| e.to_string())?;
                }
                if let Some(base) = base {
                    let metadata = json!({"version":1,"sampleRate":RATE,"channels":2,"rfSampleRate":rate,"frames":position,"points":points,"error":result.as_ref().err()});
                    let path = format!("{}.audio.json", base.display());
                    std::fs::write(path, serde_json::to_vec(&metadata).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
                }
                drop(tx);
                if let Some(worker) = playback_worker {
                    worker.join().map_err(|_| "Audio playback worker panicked")??;
                }
                println!("AUDIO_OK frames={position} monitor_dropped_blocks={monitor_drops}");
                result
            })
            .map_err(|e| e.to_string())?;
        println!("NETWORK_AUDIO rate={RATE} channels=2 recording={recording} monitor={monitor}");
        Ok(Self {
            stop,
            control,
            worker: Some(worker),
        })
    }
    pub(crate) fn finish(&mut self) -> Result<(), String> {
        // Allow capture/device latency to catch up with the last decoded field.
        thread::sleep(Duration::from_millis(200));
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.control.shutdown(Shutdown::Both);
        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| "Audio receiver panicked")??;
        }
        Ok(())
    }
}
impl Drop for Receiver {
    fn drop(&mut self) {
        if self.worker.is_some() {
            let _ = self.finish();
        }
    }
}

pub(crate) struct ExportTrack {
    path: PathBuf,
}
impl Drop for ExportTrack {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
impl ExportTrack {
    pub(crate) fn configure(track: Option<&Self>, cmd: &mut Command) {
        if let Some(track) = track {
            cmd.arg("-i").arg(&track.path).args([
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:a",
                "flac",
                "-shortest",
            ]);
        } else {
            cmd.arg("-an");
        }
    }
}
fn audio_position(points: &[(f64, f64)], rf: f64, rate: f64) -> f64 {
    if points.len() == 1 {
        return points[0].1 + (rf - points[0].0) * RATE as f64 / rate;
    }
    let upper = points
        .partition_point(|p| p.0 < rf)
        .clamp(1, points.len() - 1);
    let (a, b) = (points[upper - 1], points[upper]);
    a.1 + (rf - a.0) * (b.1 - a.1) / (b.0 - a.0)
}
/// Render each retained field's audio interval; lost/repeated fields cannot
/// silently shift all subsequent sound. Average timestamp anchors over 2 s.
pub(crate) fn export_track(
    base: &str,
    destination: &Path,
    rf_fields: &[f64],
    indices: &[usize],
    offset_ms: f64,
    cancellable: bool,
) -> Result<Option<ExportTrack>, String> {
    let base = base.strip_suffix(".tbc").unwrap_or(base);
    let wave = PathBuf::from(format!("{base}.audio.wav"));
    let timing = PathBuf::from(format!("{base}.audio.json"));
    if !wave.exists() && !timing.exists() {
        return Ok(None);
    }
    if !offset_ms.is_finite() || offset_ms.abs() > 10000.0 {
        return Err("Audio offset must be within ±10000 ms".into());
    }
    let metadata: Value = serde_json::from_reader(std::io::BufReader::new(
        File::open(&timing).map_err(|e| format!("Audio timing: {e}"))?,
    ))
    .map_err(|e| e.to_string())?;
    let rate = metadata["rfSampleRate"]
        .as_f64()
        .filter(|r| r.is_finite() && *r > 0.0)
        .ok_or("Invalid audio RF clock")?;
    if metadata["sampleRate"] != RATE || metadata["channels"] != 2 {
        return Err("Unsupported recorded audio format".into());
    }
    if !metadata["error"].is_null() {
        eprintln!(
            "AUDIO_WARNING original recording reported: {}",
            metadata["error"]
        );
    }
    let values = metadata["points"]
        .as_array()
        .ok_or("Missing audio timing points")?;
    let mut points = Vec::new();
    for chunk in values.chunks(100) {
        let (mut x, mut y) = (0.0, 0.0);
        for point in chunk {
            x += point[1]
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or("Invalid audio timestamp")?;
            y += point[0]
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or("Invalid audio sample index")?;
        }
        points.push((x / chunk.len() as f64, y / chunk.len() as f64));
    }
    if points.is_empty()
        || points
            .windows(2)
            .any(|p| p[1].0 <= p[0].0 || p[1].1 <= p[0].1)
    {
        return Err("Invalid/nonmonotonic audio timing".into());
    }
    let mut source = hound::WavReader::open(&wave).map_err(|e| format!("Recorded audio: {e}"))?;
    let spec = source.spec();
    if spec.channels != 2
        || spec.sample_rate != RATE
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        return Err("Recorded audio must be 48 kHz stereo PCM16 WAV".into());
    }
    let input_frames = source.duration() as usize;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let track = ExportTrack {
        path: destination.parent().unwrap_or(Path::new(".")).join(format!(
            ".oxide-rf-audio-{}-{stamp}.wav",
            std::process::id()
        )),
    };
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&track.path)
        .map_err(|e| e.to_string())?;
    let mut writer =
        hound::WavWriter::new(std::io::BufWriter::new(file), spec).map_err(|e| e.to_string())?;
    let mut missing = 0usize;
    for &index in indices {
        if cancellable && crate::export::cancelled() {
            return Err("Audio alignment stopped; original recording retained".into());
        }
        let rf = *rf_fields
            .get(index)
            .filter(|v| v.is_finite())
            .ok_or("Missing RF field position for audio alignment")?
            - offset_ms * rate / 1000.0;
        let begin = audio_position(&points, rf, rate);
        let end = audio_position(&points, rf + rate * 0.02, rate);
        if end <= begin || end - begin > RATE as f64 * 0.1 {
            return Err("Invalid audio clock interval".into());
        }
        let first = begin.floor().max(0.0) as usize;
        let last = (end.ceil().max(0.0) as usize + 2).min(input_frames);
        let samples = if first < last {
            source.seek(first as u32).map_err(|e| e.to_string())?;
            source
                .samples::<i16>()
                .take((last - first) * 2)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?
        } else {
            Vec::new()
        };
        for frame in 0..RATE as usize / 50 {
            let at = begin + (end - begin) * frame as f64 / (RATE as f64 / 50.0);
            let floor = at.floor();
            let fraction = at - floor;
            let local = floor as isize - first as isize;
            let available = at >= 0.0
                && at < (input_frames.saturating_sub(1)) as f64
                && local >= 0
                && (local as usize + 1) * 2 + 1 < samples.len();
            for channel in 0..2 {
                let sample = if available {
                    let a = samples[local as usize * 2 + channel] as f64;
                    let b = samples[(local as usize + 1) * 2 + channel] as f64;
                    (a + (b - a) * fraction)
                        .round()
                        .clamp(i16::MIN as f64, i16::MAX as f64) as i16
                } else {
                    0
                };
                writer.write_sample(sample).map_err(|e| e.to_string())?;
            }
            if !available {
                missing += 1;
            }
        }
    }
    writer.finalize().map_err(|e| e.to_string())?;
    println!(
        "AUDIO_ALIGN fields={} offset_ms={offset_ms} missing_frames={missing} codec=FLAC",
        indices.len()
    );
    Ok(Some(track))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blocks_convert_rate_and_channels_and_keep_capture_time() {
        // 44.1 kHz mono ramp in uneven chunks, each with its capture instant.
        let origin = Instant::now();
        let mut blocks = Blocks::new(1, 44_100);
        let mut out = Vec::new();
        let mut frame = 0usize;
        for size in [100, 4410, 333, 7000, 2157].iter().cycle().take(20) {
            let at = origin + Duration::from_secs_f64(frame as f64 / 44_100.0);
            let samples: Vec<f32> = (frame..frame + size)
                .map(|i| (i % 1000) as f32 / 2000.0)
                .collect();
            blocks
                .push(at, &samples, &mut |at, pcm: &[u8]| {
                    out.push((at, pcm.to_vec()));
                    Ok(())
                })
                .unwrap();
            frame += size;
        }
        let total = frame as f64 * 48_000.0 / 44_100.0;
        assert_eq!(out.len(), (total as usize - 1) / FRAMES);
        for (number, (at, pcm)) in out.iter().enumerate() {
            let expected = (number * FRAMES) as f64 / RATE as f64;
            assert!((at.duration_since(origin).as_secs_f64() - expected).abs() < 1e-6);
            for (k, f) in pcm.as_chunks::<4>().0.iter().enumerate() {
                let [l, r] = [
                    i16::from_le_bytes([f[0], f[1]]),
                    i16::from_le_bytes([f[2], f[3]]),
                ];
                assert_eq!(l, r);
                // Away from the ramp's wrap, interpolation follows the ramp.
                let input = (number * FRAMES + k) as f64 * 44_100.0 / 48_000.0;
                if input % 1000.0 < 999.0 {
                    let ramp = (input % 1000.0) / 2000.0 * 32767.0;
                    assert!((l as f64 - ramp).abs() <= 1.0, "{l} vs {ramp}");
                }
            }
        }
        // 48 kHz with four channels passes the first two through unchanged.
        let mut blocks = Blocks::new(4, RATE);
        let samples: Vec<f32> = (0..FRAMES * 2)
            .flat_map(|i| [i as f32 / 4096.0, -(i as f32) / 4096.0, 0.9, 0.9])
            .collect();
        let mut pcm = Vec::new();
        blocks
            .push(origin, &samples, &mut |at, block: &[u8]| {
                assert_eq!(
                    at,
                    origin + Duration::from_secs_f64(pcm.len() as f64 / 4.0 / RATE as f64)
                );
                pcm.extend_from_slice(block);
                Ok(())
            })
            .unwrap();
        assert_eq!(pcm.len(), FRAMES * 8);
        for (i, f) in pcm.as_chunks::<4>().0.iter().enumerate() {
            let value = ((i as f32 / 4096.0) * 32767.0).round() as i16;
            assert_eq!(
                [
                    i16::from_le_bytes([f[0], f[1]]),
                    i16::from_le_bytes([f[2], f[3]])
                ],
                [value, -value]
            );
        }
    }
    #[test]
    #[ignore = "records from the default audio input device"]
    fn capture_delivers_timestamped_blocks_from_the_default_input() {
        let capture = Capture::open("default").unwrap();
        let stop = AtomicBool::new(false);
        let start = Instant::now();
        let mut stamps = Vec::new();
        capture
            .run(&stop, |at, pcm| {
                assert_eq!(pcm.len(), FRAMES * 4);
                stamps.push(at);
                if stamps.len() == 100 {
                    stop.store(true, Ordering::Relaxed);
                }
                Ok(())
            })
            .unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        let span = stamps[99].duration_since(stamps[0]).as_secs_f64();
        eprintln!(
            "{} Hz × {} channels, 2 s of blocks in {elapsed:.3} s, stamp span {span:.4} s, first block {:.1} ms after open",
            capture.rate,
            capture.channels,
            stamps[0].saturating_duration_since(start).as_secs_f64() * 1000.0
        );
        assert!((span - 99.0 * FRAMES as f64 / RATE as f64).abs() < 0.05);
    }
    #[test]
    #[ignore = "opens the default audio output device"]
    fn monitor_plays_on_the_default_output() {
        let mut monitor = Monitor::open().unwrap();
        // 0.3 s of silence at 48 kHz stereo.
        monitor.push(&vec![0u8; 48_000 * 4 * 3 / 10]);
        thread::sleep(Duration::from_millis(400));
        assert!(monitor.buffer.lock().unwrap().len() < monitor.limit);
    }
    #[test]
    fn receiver_preserves_pcm_and_reports_missing_end_marker_or_truncated_payload() {
        for scenario in 0..3 {
            let dir = std::env::temp_dir().join(format!(
                "vhs-audio-wire-{}-{}-{scenario}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let base = dir.join("capture");
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let server = thread::spawn(move || {
                let (mut socket, _) = listener.accept().unwrap();
                let mut hello = [0u8; 16];
                socket.read_exact(&mut hello).unwrap();
                assert_eq!(&hello[..8], MAGIC);
                assert_eq!(u64::from_le_bytes(hello[8..].try_into().unwrap()), 123);
                socket.write_all(MAGIC).unwrap();
                let mut header = [0u8; 20];
                header[..8].copy_from_slice(&1000f64.to_le_bytes());
                header[16..].copy_from_slice(&2u32.to_le_bytes());
                socket.write_all(&header).unwrap();
                socket.write_all(&[1, 0, 2, 0, 3, 0, 4, 0]).unwrap();
                if scenario == 0 {
                    header[8..16].copy_from_slice(&2u64.to_le_bytes());
                    header[16..].fill(0);
                    socket.write_all(&header).unwrap();
                } else if scenario == 2 {
                    header[8..16].copy_from_slice(&2u64.to_le_bytes());
                    socket.write_all(&header).unwrap();
                    socket.write_all(&[5, 0]).unwrap();
                }
            });
            let mut receiver =
                Receiver::connect("127.0.0.1:5555", port, 123, Some(&base), false, 28e6).unwrap();
            server.join().unwrap();
            let deadline = Instant::now() + Duration::from_secs(2);
            while !receiver.worker.as_ref().unwrap().is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            assert!(receiver.worker.as_ref().unwrap().is_finished());
            let result = receiver.finish();
            let metadata: Value =
                serde_json::from_slice(&std::fs::read(dir.join("capture.audio.json")).unwrap())
                    .unwrap();
            assert_eq!(metadata["frames"], 2);
            let samples = hound::WavReader::open(dir.join("capture.audio.wav"))
                .unwrap()
                .into_samples::<i16>()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(samples, [1, 2, 3, 4]);
            if scenario == 0 {
                result.unwrap();
                assert!(metadata["error"].is_null());
            } else {
                assert!(result.is_err());
                assert!(metadata["error"].is_string());
            }
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
    #[test]
    fn alignment_compensates_clock_drift_skipped_fields_and_user_offset() {
        let dir = std::env::temp_dir().join(format!(
            "vhs-audio-align-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("capture");
        let base = base.to_str().unwrap();
        let mut wav = hound::WavWriter::create(
            format!("{base}.audio.wav"),
            hound::WavSpec {
                channels: 2,
                sample_rate: RATE,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for i in 0..20000i16 {
            wav.write_sample(i).unwrap();
            wav.write_sample(-i).unwrap();
        }
        wav.finalize().unwrap();
        let rf_rate = 28e6;
        let drift = 1.01;
        let points: Vec<Value> = (0..300)
            .map(|i| json!([i * 48, i as f64 * 48.0 / (RATE as f64 * drift) * rf_rate]))
            .collect();
        std::fs::write(
            format!("{base}.audio.json"),
            serde_json::to_vec(
                &json!({"sampleRate":RATE,"channels":2,"rfSampleRate":rf_rate,"points":points}),
            )
            .unwrap(),
        )
        .unwrap();
        let fields = vec![0.0, rf_rate * 0.02, rf_rate * 0.04, rf_rate * 0.06];
        let track = export_track(base, &dir.join("test.mkv"), &fields, &[1, 3], 10.0, true)
            .unwrap()
            .unwrap();
        let mut reader = hound::WavReader::open(&track.path).unwrap();
        assert_eq!(reader.duration(), 1920);
        let samples = reader
            .samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!((samples[0] as i32 - 485).abs() <= 1);
        assert_eq!(samples[1], -samples[0]);
        assert!(
            (samples[1920] as i32 - 2424).abs() <= 1,
            "Skipped RF field must skip matching audio"
        );
        drop(track);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
