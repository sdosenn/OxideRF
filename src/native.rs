//! Experimental native PAL VHS RF decoder. No interpreter or Python runtime.
//! Deliberately narrower than VHS-Decode: SP PAL, basic sync/TBC and burst AGC.
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

pub(crate) const N: usize = 32768;
pub(crate) const EDGE: usize = 2048;
pub(crate) const HOP: usize = N - 2 * EDGE;
pub(crate) const WIDTH: usize = 1135;
pub(crate) const HEIGHT: usize = 313;
const FSC: f64 = 4_433_618.75;
const OUT_RATE: f64 = 4.0 * FSC;
pub(crate) const UNDER: f64 = 626_953.125;
const PAL_SYNC_SLICE_HZ: f32 = 3.95e6;
static STOP: AtomicBool = AtomicBool::new(false);
pub(crate) fn stop_requested() -> bool {
    STOP.load(Ordering::Relaxed)
}
/// Environment variable set by the GUI: its jobs read `stop` lines from stdin,
/// which ends them gracefully on every platform (Windows has no SIGINT to send).
pub const STDIN_CONTROL: &str = "OXIDE_RF_STDIN_CONTROL";

/// Stop gracefully on Ctrl+C/SIGTERM and, for GUI jobs, on a `stop` line.
pub(crate) fn install_stop_handler() -> Result<(), String> {
    static INSTALLED: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    STOP.store(false, Ordering::Relaxed);
    INSTALLED
        .get_or_init(|| {
            ctrlc::set_handler(|| STOP.store(true, Ordering::Relaxed))
                .map_err(|e| format!("Stop handler: {e}"))?;
            if std::env::var_os(STDIN_CONTROL).is_some() {
                std::thread::spawn(|| {
                    for line in std::io::stdin().lines().map_while(Result::ok) {
                        if line.trim() == "stop" {
                            STOP.store(true, Ordering::Relaxed);
                        }
                    }
                });
            }
            Ok(())
        })
        .clone()
}

#[derive(Clone, Copy, Default)]
pub(crate) struct C {
    pub(crate) re: f32,
    pub(crate) im: f32,
}
impl C {
    fn mul(self, b: Self) -> Self {
        Self {
            re: self.re * b.re - self.im * b.im,
            im: self.re * b.im + self.im * b.re,
        }
    }
    fn scale(self, x: f32) -> Self {
        Self {
            re: self.re * x,
            im: self.im * x,
        }
    }
    pub(crate) fn polar(a: f64) -> Self {
        Self {
            re: a.cos() as f32,
            im: a.sin() as f32,
        }
    }
}
struct PalFilters {
    rate: f64,
    video: Vec<C>,
    sync: Vec<f32>,
}
struct Fft {
    filters: std::cell::RefCell<Option<PalFilters>>,
    gpu: bool,
    twiddle: Vec<C>,
}
impl Fft {
    fn new(gpu: bool) -> Self {
        Self {
            filters: std::cell::RefCell::new(None),
            gpu,
            twiddle: (0..N / 2)
                .map(|i| C::polar(-std::f64::consts::TAU * i as f64 / N as f64))
                .collect(),
        }
    }
    fn pal_filters(&self, rate: f64) -> std::cell::Ref<'_, PalFilters> {
        if self
            .filters
            .borrow()
            .as_ref()
            .is_none_or(|f| f.rate != rate)
        {
            let video = (0..N)
                .map(|k| {
                    let signed = if k <= N / 2 {
                        k as f64
                    } else {
                        k as f64 - N as f64
                    };
                    pal_video_gain(signed * rate / N as f64, rate)
                })
                .collect();
            let sync = (0..N)
                .map(|k| pal_sync_gain(k.min(N - k) as f64 * rate / N as f64))
                .collect();
            *self.filters.borrow_mut() = Some(PalFilters { rate, video, sync });
        }
        std::cell::Ref::map(self.filters.borrow(), |v| v.as_ref().unwrap())
    }
    fn run(&self, x: &mut [C], inverse: bool) -> Result<(), String> {
        if self.gpu {
            let input: Vec<f32> = x.iter().flat_map(|c| [c.re, c.im]).collect();
            let mut output = vec![0f32; 2 * N];
            if unsafe { crate::vhs_fft(input.as_ptr(), output.as_mut_ptr(), N, inverse as i32) }
                != 0
            {
                let p = crate::vhs_error();
                let error = unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() };
                unsafe {
                    crate::vhs_free_error(p);
                }
                return Err(error);
            }
            for (c, v) in x.iter_mut().zip(output.as_chunks::<2>().0) {
                *c = C { re: v[0], im: v[1] };
            }
        } else {
            for i in 0..N {
                let j = i.reverse_bits() >> (usize::BITS - N.trailing_zeros());
                if j > i {
                    x.swap(i, j);
                }
            }
            for bits in 1..=N.trailing_zeros() {
                let width = 1 << bits;
                for base in (0..N).step_by(width) {
                    for j in 0..width / 2 {
                        let mut w = self.twiddle[j * N / width];
                        if inverse {
                            w.im = -w.im;
                        }
                        let u = x[base + j];
                        let t = x[base + j + width / 2].mul(w);
                        x[base + j] = C {
                            re: u.re + t.re,
                            im: u.im + t.im,
                        };
                        x[base + j + width / 2] = C {
                            re: u.re - t.re,
                            im: u.im - t.im,
                        };
                    }
                }
            }
            if inverse {
                for c in x {
                    *c = c.scale(1.0 / N as f32);
                }
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct LiveCounters {
    skipped: usize,
    estimated: usize,
    input_wait: std::time::Duration,
}
impl LiveCounters {
    fn sample(
        &self,
        source: &Source,
        output: &Output,
        demod: std::time::Duration,
        field: std::time::Duration,
    ) -> crate::diagnostics::Sample {
        let (received, queued, capacity) = match source {
            Source::Live(stream) => stream.levels(),
            _ => (0, 0, 0),
        };
        let direct = output.direct.as_ref();
        let mkv = |read: fn(&crate::export::DirectStats) -> u64| {
            direct.map_or(0, |direct| read(direct.stats()))
        };
        crate::diagnostics::Sample {
            fields: output.count,
            skipped: self.skipped,
            estimated: self.estimated,
            dropout_fields: output.dropout_fields,
            demod,
            field,
            input_wait: self.input_wait,
            received,
            queued,
            capacity,
            mkv_queue: direct.map_or(0, |direct| direct.queued()),
            mkv_send_ns: mkv(|s| s.send_ns.load(Ordering::Relaxed)),
            mkv_render_ns: mkv(|s| s.render_ns.load(Ordering::Relaxed)),
            mkv_write_ns: mkv(|s| s.write_ns.load(Ordering::Relaxed)),
        }
    }
}

enum Source {
    Live(crate::cxadc::Stream),
    Pcm {
        reader: BufReader<File>,
        kind: String,
        bytes: usize,
    },
    Flac {
        reader: claxon::FlacReader<File>,
        pending: VecDeque<f32>,
    },
    Wav(hound::WavReader<BufReader<File>>),
}
impl Source {
    fn open(path: &Path, kind: &str) -> Result<Self, String> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        let kind = if kind == "auto" {
            match ext.as_str() {
                "r8" | "u8" => "u8",
                "r16" | "u16" => "u16",
                "s8" => "s8",
                "s16" => "s16",
                "rf" => "f32",
                "flac" | "ldf" | "vhs" => "flac",
                "wav" => "wav",
                _ => return Err("For RAW, select u8/u16/s16/s8/f32".into()),
            }
        } else {
            kind
        };
        match kind {
            "flac" => {
                let f = claxon::FlacReader::open(path).map_err(|e| e.to_string())?;
                if f.streaminfo().channels != 1 {
                    return Err("RF FLAC must be mono".into());
                }
                Ok(Self::Flac {
                    reader: f,
                    pending: VecDeque::new(),
                })
            }
            "wav" => {
                let f = hound::WavReader::open(path).map_err(|e| e.to_string())?;
                if f.spec().channels != 1 {
                    return Err("RF WAV must be mono".into());
                }
                Ok(Self::Wav(f))
            }
            "u8" | "s8" | "u16" | "s16" | "f32" => Ok(Self::Pcm {
                reader: BufReader::new(File::open(path).map_err(|e| e.to_string())?),
                bytes: match kind {
                    "u8" | "s8" => 1,
                    "f32" => 4,
                    _ => 2,
                },
                kind: kind.into(),
            }),
            _ => Err("Unsupported sample format".into()),
        }
    }
    fn read(&mut self, count: usize) -> Result<Vec<f32>, String> {
        match self {
            Self::Live(stream) => stream.read(count),
            Self::Pcm {
                reader,
                kind,
                bytes,
            } => {
                let mut data = vec![0u8; count * *bytes];
                let mut len = 0;
                while len < data.len() {
                    let n = reader.read(&mut data[len..]).map_err(|e| e.to_string())?;
                    if n == 0 {
                        break;
                    }
                    len += n;
                }
                if len % *bytes != 0 {
                    return Err("Incomplete PCM sample at end of file".into());
                }
                Ok(data[..len]
                    .chunks_exact(*bytes)
                    .map(|b| match kind.as_str() {
                        "u8" => (b[0] as f32 - 128.0) / 128.0,
                        "s8" => (b[0] as i8) as f32 / 128.0,
                        "u16" => (u16::from_le_bytes([b[0], b[1]]) as f32 - 32768.0) / 32768.0,
                        "s16" => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
                        _ => f32::from_le_bytes(b.try_into().unwrap()),
                    })
                    .collect())
            }
            Self::Flac { reader, pending } => {
                let scale = 2f32.powi(reader.streaminfo().bits_per_sample as i32 - 1);
                let mut result = Vec::with_capacity(count);
                while result.len() < count {
                    if pending.is_empty() {
                        let block = reader
                            .blocks()
                            .read_next_or_eof(Vec::new())
                            .map_err(|e| e.to_string())?;
                        let Some(block) = block else {
                            break;
                        };
                        pending.extend(block.channel(0).iter().map(|&x| x as f32 / scale));
                    }
                    let n = (count - result.len()).min(pending.len());
                    result.extend(pending.drain(..n));
                }
                Ok(result)
            }
            Self::Wav(f) => {
                let spec = f.spec();
                if spec.sample_format == hound::SampleFormat::Float {
                    f.samples::<f32>()
                        .take(count)
                        .map(|s| s.map_err(|e| e.to_string()))
                        .collect()
                } else {
                    let scale = 2f32.powi(spec.bits_per_sample as i32 - 1);
                    f.samples::<i32>()
                        .take(count)
                        .map(|s| s.map(|x| x as f32 / scale).map_err(|e| e.to_string()))
                        .collect()
                }
            }
        }
    }
    fn skip(&mut self, samples: u64) -> Result<(), String> {
        if samples == 0 {
            return Ok(());
        }
        if matches!(self, Self::Live(_)) {
            return Err("Live RF input cannot seek to a start frame".into());
        }
        if let Self::Pcm { reader, bytes, .. } = self {
            reader
                .seek(SeekFrom::Start(samples * *bytes as u64))
                .map_err(|e| e.to_string())?;
        } else {
            let mut remaining = samples;
            while remaining > 0 {
                let n = remaining.min(1 << 20) as usize;
                let got = self.read(n)?.len();
                if got == 0 {
                    break;
                }
                remaining -= got as u64;
            }
        }
        Ok(())
    }
}

pub(crate) fn lowpass(f: f64, cutoff: f64, transition: f64) -> f32 {
    if f <= cutoff {
        1.0
    } else if f >= cutoff + transition {
        0.0
    } else {
        (0.5 + 0.5 * (std::f64::consts::PI * (f - cutoff) / transition).cos()) as f32
    }
}

pub(crate) fn pal_rf_gain(f: f64, fs: f64) -> f32 {
    if f <= UNDER || f >= fs / 2.0 {
        return 0.0;
    }
    let warped = (std::f64::consts::PI * f / fs).tan();
    let high = (std::f64::consts::PI * 1.32e6 / fs).tan() / warped;
    let low = warped / (std::f64::consts::PI * 5.91e6 / fs).tan();
    let band = 1.0 / ((1.0 + high.powi(24)) * (1.0 + low.powi(40))).sqrt();
    // VHS playback RF needs rising equalization before the FM discriminator.
    // A flat RF passband leaves severe sideband distortion on picture detail.
    let r = warped / (std::f64::consts::PI * 4.3e6 / fs).tan();
    // PAL VHS RF peaking from the reference profile: 4 dB at 4.3 MHz.
    // Apply before FM discrimination rather than sharpening noisy decoded pixels.
    let width = std::f64::consts::LN_2 * 10e6 / fs;
    // Use exp rather than the host's newly versioned sinh symbol so the
    // portable binary keeps the Zorin 18 / glibc 2.39 compatibility baseline.
    let q = 1.0 / (width.exp() - (-width).exp());
    let a = 10f64.powf(4.0 / 20.0);
    let d = (1.0 - r * r).powi(2);
    let peak = ((d + (a * r / q).powi(2)) / (d + (r / q).powi(2))).sqrt();
    (2.0 * band * peak * (f - UNDER) / (4.3e6 - UNDER)) as f32
}
/// PAL VHS playback de-emphasis: inverse high shelf, followed by the
/// sixth-order 3.4 MHz video low-pass. Keep CPU and OpenCL coefficients identical.
/// Response follows VHS-Decode's FMDeEmphasisB; see THIRD_PARTY_NOTICES.md.
pub(crate) fn pal_video_gain(f: f64, fs: f64) -> C {
    let a = 10f64.powf(13.9794 / 40.0);
    let w0 = std::f64::consts::TAU * 273_755.82 / fs;
    let alpha = w0.sin() / (2.0 * 0.462_088_186);
    let c = w0.cos();
    let t = 2.0 * a.sqrt() * alpha;
    let numerator = [
        (a + 1.0) - (a - 1.0) * c + t,
        2.0 * ((a - 1.0) - (a + 1.0) * c),
        (a + 1.0) - (a - 1.0) * c - t,
    ];
    let denominator = [
        a * ((a + 1.0) + (a - 1.0) * c + t),
        -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
        a * ((a + 1.0) + (a - 1.0) * c - t),
    ];
    let w = std::f64::consts::TAU * f / fs;
    let response = |v: [f64; 3]| {
        (
            v[0] + v[1] * w.cos() + v[2] * (2.0 * w).cos(),
            -v[1] * w.sin() - v[2] * (2.0 * w).sin(),
        )
    };
    let (nr, ni) = response(numerator);
    let (dr, di) = response(denominator);
    let power = dr * dr + di * di;
    let warped =
        (std::f64::consts::PI * f.abs() / fs).tan() / (std::f64::consts::PI * 3.4e6 / fs).tan();
    let low = 1.0 / (1.0 + warped.powi(12)).sqrt();
    C {
        re: ((nr * dr + ni * di) / power * low) as f32,
        im: ((ni * dr - nr * di) / power * low) as f32,
    }
}

/// Zero-phase 65-tap Hamming sync FIR, expressed in seconds at the reference
/// 40 MHz design clock so its bandwidth remains fixed at every RF input rate.
pub(crate) fn pal_sync_gain(f: f64) -> f32 {
    let mut sum = 0.0;
    let mut dc = 0.0;
    for n in -32..=32 {
        let x = 0.025 * n as f64;
        let sinc = if n == 0 {
            1.0
        } else {
            (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
        };
        let tap = 0.025 * sinc * (0.54 + 0.46 * (std::f64::consts::PI * n as f64 / 32.0).cos());
        dc += tap;
        sum += tap * (std::f64::consts::TAU * f * n as f64 / 40e6).cos();
    }
    (sum / dc) as f32
}
struct Samples {
    luma: Vec<f32>,
    envelope: Vec<f32>,
    sync: Vec<f32>,
    chroma: Vec<C>,
    base: u64,
    resident_len: Option<usize>,
}
impl Samples {
    fn len(&self) -> usize {
        self.resident_len.unwrap_or(self.luma.len())
    }
    fn trim(&mut self, n: usize) {
        if let Some(len) = &mut self.resident_len {
            *len -= n;
            self.base += n as u64;
            return;
        }
        self.luma.drain(..n);
        self.envelope.drain(..n);
        self.sync.drain(..n);
        self.chroma.drain(..n);
        self.base += n as u64;
    }
}
fn demod(raw: &[f32], fs: f64, fft: &Fft, absolute: u64) -> Result<Samples, String> {
    demod_mode(raw, fs, fft, absolute, false)
}
fn demod_mode(
    raw: &[f32],
    fs: f64,
    fft: &Fft,
    absolute: u64,
    luma_only: bool,
) -> Result<Samples, String> {
    let mean = raw.iter().sum::<f32>() / N as f32;
    let mut spectrum: Vec<C> = raw
        .iter()
        .map(|s| C {
            re: s - mean,
            im: 0.0,
        })
        .collect();
    fft.run(&mut spectrum, false)?;
    let mut analytic = spectrum.clone();
    for (k, x) in analytic.iter_mut().enumerate() {
        let f = k as f64 * fs / N as f64;
        let gain = if k < N / 2 { pal_rf_gain(f, fs) } else { 0.0 };
        *x = x.scale(gain);
    }
    fft.run(&mut analytic, true)?;
    let mut freq = vec![C::default(); N];
    for i in 1..N {
        let a = analytic[i - 1];
        let b = analytic[i];
        let phase = (b.im * a.re - b.re * a.im)
            .atan2(b.re * a.re + b.im * a.im)
            .rem_euclid(std::f32::consts::TAU);
        freq[i].re = (phase as f64 * fs / std::f64::consts::TAU) as f32;
    }
    freq[0] = freq[1];
    fft.run(&mut freq, false)?;
    // Reuse the frequency-domain filters across RF blocks.
    let filters = fft.pal_filters(fs);
    for (x, gain) in freq.iter_mut().zip(&filters.video) {
        *x = x.mul(*gain);
    }
    let mut sync = freq.clone();
    for (x, gain) in sync.iter_mut().zip(&filters.sync) {
        *x = x.scale(*gain);
    }
    fft.run(&mut sync, true)?;
    fft.run(&mut freq, true)?;
    if !luma_only {
        for (k, x) in spectrum.iter_mut().enumerate() {
            let f = k as f64 * fs / N as f64;
            *x = x.scale(if k < N / 2 {
                2.0 * (1.0 - lowpass(f, 0.15e6, 0.1e6)) * lowpass(f, 1.0e6, 0.15e6)
            } else {
                0.0
            });
        }
        fft.run(&mut spectrum, true)?;
    }
    let chroma = if luma_only {
        vec![C::default(); HOP]
    } else {
        (EDGE..N - EDGE)
            .map(|i| {
                spectrum[i].mul(C::polar(
                    -std::f64::consts::TAU * UNDER * (absolute + i as u64) as f64 / fs,
                ))
            })
            .collect()
    };
    Ok(Samples {
        luma: freq[EDGE..N - EDGE].iter().map(|c| c.re).collect(),
        envelope: analytic[EDGE..N - EDGE]
            .iter()
            .map(|c| c.re.hypot(c.im))
            .collect(),
        sync: sync[EDGE..N - EDGE].iter().map(|c| c.re).collect(),
        chroma,
        base: absolute + EDGE as u64,
        resident_len: None,
    })
}

#[derive(Default)]
struct PulseFinder {
    low: Option<u64>,
    last_broad: Option<u64>,
    broad_count: usize,
    broad_start: u64,
    horizontal: Vec<u64>,
    vertical: Vec<u64>,
    recovered: std::collections::BTreeSet<u64>,
    clock_fraction: f64,
}
fn adaptive_sync_slice(values: &[f32]) -> f32 {
    let mut histogram = [0usize; 128];
    for &v in values {
        let bin = ((v - 3.2e6) / 16000.0).floor().clamp(0.0, 127.0) as usize;
        histogram[bin] += 1;
    }
    let target = (values.len() * 3 / 100).max(1);
    let mut count = 0;
    for (bin, n) in histogram.into_iter().enumerate() {
        count += n;
        if count >= target {
            return (3.2e6 + (bin as f32 + 0.5) * 16000.0 + 150000.0).clamp(3.65e6, 4.4e6);
        }
    }
    PAL_SYNC_SLICE_HZ
}
impl PulseFinder {
    fn push(&mut self, values: &[f32], base: u64, fs: f64) {
        let slice = adaptive_sync_slice(values);
        for (i, &value) in values.iter().enumerate() {
            let pos = base + i as u64;
            if value < slice {
                if self.low.is_none() {
                    self.low = Some(pos);
                }
            } else if let Some(start) = self.low.take() {
                self.pulse(start, pos - start, fs);
            }
        }
    }
    fn maintain_field_clock(&mut self, end: u64, fs: f64) {
        let period = fs * 0.02;
        let mut index = 1;
        while index < self.vertical.len() {
            let a = self.vertical[index - 1];
            let b = self.vertical[index];
            let steps = ((b - a) as f64 / period).round() as usize;
            if steps > 1 {
                let missing: Vec<_> = (1..steps)
                    .map(|i| a + ((b - a) as f64 * i as f64 / steps as f64).round() as u64)
                    .collect();
                self.recovered.extend(missing.iter().copied());
                self.vertical.splice(index..index, missing);
                index += steps;
            } else {
                index += 1;
            }
        }
        while let Some(&last) = self.vertical.last() {
            if end.saturating_sub(last) as f64 <= period + fs * 0.001 {
                break;
            }
            let step = (period + self.clock_fraction).round() as u64;
            self.clock_fraction += period - step as f64;
            let next = last + step;
            self.recovered.insert(next);
            self.vertical.push(next);
        }
    }
    fn pulse(&mut self, start: u64, length: u64, fs: f64) {
        let us = length as f64 * 1e6 / fs;
        if (3.0..7.0).contains(&us) {
            self.horizontal.push(start);
        }
        if (20.0..30.5).contains(&us) {
            if self
                .last_broad
                .is_some_and(|p| (25.0..40.0).contains(&((start - p) as f64 * 1e6 / fs)))
            {
                self.broad_count += 1;
            } else {
                self.broad_count = 1;
                self.broad_start = start;
            }
            self.last_broad = Some(start);
            if self.broad_count == 4 {
                let origin = self
                    .broad_start
                    .saturating_sub((160e-6 * fs).round() as u64);
                if self
                    .vertical
                    .last()
                    .is_none_or(|&v| origin > v + (0.015 * fs) as u64)
                {
                    self.vertical.push(origin);
                    self.clock_fraction = 0.0;
                }
            }
        }
    }
}

fn interpolate(v: &[f32], p: f64) -> f32 {
    let i = (p.floor().max(0.0) as usize).min(v.len() - 2);
    v[i] + (v[i + 1] - v[i]) * (p - i as f64).clamp(0.0, 1.0) as f32
}
fn cubic(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    b + 0.5 * t * (c - a + t * (2.0 * a - 5.0 * b + 4.0 * c - d + t * (3.0 * (b - c) + d - a)))
}
fn interpolate_video(v: &[f32], p: f64) -> f32 {
    crate::resample::sample(v, p)
}

fn interpolate_c(v: &[C], p: f64) -> C {
    let i = (p.floor().max(0.0) as usize).min(v.len() - 2);
    let t = (p - i as f64).clamp(0.0, 1.0) as f32;
    C {
        re: v[i].re + (v[i + 1].re - v[i].re) * t,
        im: v[i].im + (v[i + 1].im - v[i].im) * t,
    }
}

fn restore_chroma(under: C, phase: f64, target: f64) -> C {
    C {
        re: under.re,
        im: -under.im,
    }
    .mul(C::polar(target + phase))
}
fn pal_chroma_flip(bursts: &[(f64, f32)]) -> bool {
    let mut track = 0.0;
    for row in 18..bursts.len().saturating_sub(10) {
        if bursts[row].1 > 0.0 && bursts[row - 2].1 > 0.0 {
            track += (bursts[row].0 - bursts[row - 2].0).cos();
        }
    }
    let rotated = track < 0.0;
    let mut polarity = 0.0;
    for row in 18..bursts.len().saturating_sub(10) {
        if bursts[row].1 > 0.0 && bursts[row - 1].1 > 0.0 {
            let delta = bursts[row].0 - bursts[row - 1].0;
            polarity += if row % 2 == 0 { 1.0 } else { -1.0 }
                * if rotated { delta.cos() } else { delta.sin() };
        }
    }
    polarity < 0.0
}
fn pal_burst_phase(row: usize) -> f64 {
    // Four-fsc PAL advances -90 degrees per 1135-sample line. Omitting this
    // makes lines two apart identical, so a PAL comb filter cancels all chroma.
    if row % 4 < 2 {
        135f64.to_radians()
    } else {
        315f64.to_radians()
    }
}
fn smooth_chroma_gains(bursts: &mut [(f64, f32)]) {
    let raw: Vec<f32> = bursts.iter().map(|b| b.1).collect();
    let mut valid: Vec<f32> = raw
        .iter()
        .skip(18)
        .take(HEIGHT - 28)
        .copied()
        .filter(|g| *g > 0.0)
        .collect();
    if valid.is_empty() {
        return;
    }
    let mid = median(&mut valid);
    let mut deviations: Vec<_> = valid.iter().map(|g| (*g - mid).abs()).collect();
    let cap = mid + 2.0 * median(&mut deviations);
    for (row, burst) in bursts.iter_mut().enumerate() {
        if raw[row] <= 0.0 {
            continue;
        }
        let mut sum = 0.0;
        let mut count = 0;
        for &gain in &raw[row.saturating_sub(4)..(row + 5).min(raw.len())] {
            if gain > 0.0 {
                sum += gain.min(cap);
                count += 1;
            }
        }
        burst.1 = sum / count as f32;
    }
}
fn reject_sync_outliers(locations: &[f64], locked: &mut [bool], fs: f64) {
    let line = 64e-6 * fs;
    let good = locked.to_vec();
    for row in 12..locations.len().saturating_sub(4) {
        if !good[row] {
            continue;
        }
        let mut local: Vec<f32> = (row - 3..=row + 3)
            .filter(|&i| i != row && good[i])
            .map(|i| (locations[i] - i as f64 * line - locations[0]) as f32)
            .collect();
        if local.len() < 4 {
            continue;
        }
        let center = median(&mut local);
        let mut deviations: Vec<_> = local.iter().map(|v| (*v - center).abs()).collect();
        let tolerance = (0.18e-6 * fs).max(6.0 * median(&mut deviations) as f64);
        let offset = locations[row] - row as f64 * line - locations[0];
        if (offset - center as f64).abs() > tolerance {
            locked[row] = false;
        }
    }
}
fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

fn sync_levels(sync: &[f32], edge: f64, fs: f64) -> (f32, f32) {
    let mut tip: Vec<_> = (0..8)
        .map(|i| interpolate(sync, edge + (0.8 + i as f64 * 0.18) * 1e-6 * fs))
        .collect();
    let mut porch: Vec<_> = (0..8)
        .map(|i| interpolate(sync, edge + (6.8 + i as f64 * 0.2) * 1e-6 * fs))
        .collect();
    (median(&mut tip), median(&mut porch))
}

#[cfg(test)]
fn refine_hsync(sync: &[f32], edge: f64, fs: f64) -> f64 {
    refine_hsync_at_level(sync, edge, fs, None).unwrap_or(edge)
}

fn refine_hsync_at_level(sync: &[f32], edge: f64, fs: f64, threshold: Option<f32>) -> Option<f64> {
    let (tip, porch) = sync_levels(sync, edge, fs);
    if !(100_000.0..600_000.0).contains(&(porch - tip)) {
        return None;
    }
    let level = threshold.unwrap_or((tip + porch) * 0.5);
    let start = (edge - 3e-6 * fs).max(1.0) as usize;
    let end = ((edge + 1e-6 * fs) as usize).min(sync.len() - 2);
    let mut best = edge;
    let mut distance = f64::INFINITY;
    for i in start..end {
        if sync[i - 1] > level && sync[i] <= level {
            let t = (sync[i - 1] - level) / (sync[i - 1] - sync[i]);
            let pos = (i - 1) as f64 + t as f64;
            if (pos - edge).abs() < distance {
                best = pos;
                distance = (pos - edge).abs();
            }
        }
    }
    if !distance.is_finite() {
        return None;
    }
    // Prefer the trailing edge minus the standard PAL 4.7 us sync width.
    // The reference decoder's 2.25-at-40-MHz phase correction is a time,
    // so scale it to the cxadc input clock instead of using fixed samples.
    let start = ((edge + 3.0e-6 * fs) as usize).min(sync.len() - 2);
    let end = ((edge + 6.5e-6 * fs) as usize).min(sync.len() - 2);
    // A common field threshold prevents porch noise from moving every line's
    // crossing independently. Initial measurements still use local midpoints.
    let right_level = level;
    for i in start..end {
        if sync[i - 1] < right_level && sync[i] >= right_level {
            let t = (right_level - sync[i - 1]) / (sync[i] - sync[i - 1]);
            let right = (i - 1) as f64 + t as f64;
            // The raster phase is in four-fSC samples, independent of RF rate.
            // Three samples retain PAL registration with the matched filters.
            let from_right = right - (4.7e-6 - 2.25 / 40e6 + 3.0 / OUT_RATE) * fs;
            if ((right - best) / fs - 4.7e-6).abs() < 0.7e-6 {
                return Some(from_right);
            }
        }
    }
    None
}

fn repair_line_locations(locations: &mut [f64], locked: &[bool], nominal: f64) {
    // Use measured timing at both ends of a gap. At field boundaries retain
    // the nearest measured phase and local period, including RF clock error.
    let good: Vec<_> = (0..locations.len()).filter(|&i| locked[i]).collect();
    let Some(&first) = good.first() else {
        return;
    };
    let last = *good.last().unwrap();
    let period = |indices: &[usize]| {
        let mut periods: Vec<_> = indices
            .windows(2)
            .map(|p| ((locations[p[1]] - locations[p[0]]) / (p[1] - p[0]) as f64) as f32)
            .filter(|p| (*p as f64 - nominal).abs() < nominal * 0.01)
            .collect();
        if periods.is_empty() {
            nominal
        } else {
            median(&mut periods) as f64
        }
    };
    let leading_period = period(&good[..good.len().min(8)]);
    let trailing_period = period(&good[good.len().saturating_sub(8)..]);
    for row in 0..first {
        locations[row] = locations[first] - (first - row) as f64 * leading_period;
    }
    for pair in good.windows(2) {
        let [a, b] = [pair[0], pair[1]];
        for row in a + 1..b {
            locations[row] =
                locations[a] + (locations[b] - locations[a]) * (row - a) as f64 / (b - a) as f64;
        }
    }
    for row in last + 1..locations.len() {
        locations[row] = locations[last] + (row - last) as f64 * trailing_period;
    }
}

fn pal_field_origin(anchor: f64, pre_equalization: f64, line: f64) -> (f64, bool) {
    let relative = (anchor - pre_equalization) / line;
    // PAL first-field pre-equalization is half-line aligned to H; second is
    // full-line aligned. Rounding a half integer made this jump a row on noise.
    let first = (relative - relative.round()).abs() > 0.25;
    // ld/VHS-decode row 0 is the first FULL broad-sync line, not the beginning
    // of the five pre-equalization half-line pulses.
    let full_sync = pre_equalization + if first { 2.5 * line } else { 3.0 * line };
    let origin = anchor - ((anchor - full_sync) / line).round() * line;
    (origin, first)
}

struct Output {
    direct: Option<crate::export::DirectMkvWorker>,
    y: Option<BufWriter<File>>,
    c: Option<BufWriter<File>>,
    base: PathBuf,
    fields: Vec<Value>,
    preview: Option<PathBuf>,
    checkpoint: std::time::Instant,
    preview_updated: Option<std::time::Instant>,
    last_preview: Option<(Vec<u16>, u64)>,
    previous_preview: Option<(Vec<u16>, u64)>,
    monitor_assembler: crate::preview::FrameAssembler,
    socket: Option<crate::channel::PreviewSender>,
    monitor: bool,
    count: usize,
    last_first: Option<bool>,
    adaptive_black: bool,
    colour_preview: bool,
    colour_cpu: Option<crate::export::Cpu>,
    last_colour_preview: Option<crate::preview::PreviewFrame>,
    colour_pair_pending: bool,
    /// Preview the full TBC raster instead of the active 4:3 picture.
    full_raster: bool,
    raster_chroma: crate::preview::RasterChroma,
    /// Whether a teletext header image has been received from the VBI client.
    teletext_headers: bool,
    /// Columns of each colour preview field (720, or the full-raster width).
    colour_width: usize,
    vbi: Option<crate::vbi::Server>,
    dropouts: bool,
    dropout_threshold: f32,
    dropout_fields: usize,
}
impl Output {
    fn open(
        base: &Path,
        preview: Option<PathBuf>,
        monitor: bool,
        socket: Option<crate::channel::PreviewSender>,
        direct: Option<crate::export::DirectMkvWorker>,
    ) -> Result<Self, String> {
        let tbc = !monitor && direct.is_none();
        if tbc {
            if let Some(p) = base.parent() {
                fs::create_dir_all(p).map_err(|e| e.to_string())?;
            }
            for suffix in [".tbc", "_chroma.tbc", ".tbc.json"] {
                if PathBuf::from(format!("{}{suffix}", base.display())).exists() {
                    return Err(format!("Output already exists: {}{suffix}", base.display()));
                }
            }
        }
        let open = |s| {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(format!("{}{s}", base.display()))
                .map(BufWriter::new)
                .map_err(|e| e.to_string())
        };
        Ok(Self {
            y: if tbc { Some(open(".tbc")?) } else { None },
            c: if tbc {
                Some(open("_chroma.tbc")?)
            } else {
                None
            },
            direct,
            base: base.into(),
            fields: Vec::new(),
            preview,
            checkpoint: std::time::Instant::now(),
            preview_updated: None,
            last_preview: None,
            previous_preview: None,
            monitor_assembler: Default::default(),
            socket,
            monitor,
            count: 0,
            last_first: None,
            adaptive_black: false,
            colour_preview: false,
            colour_cpu: None,
            last_colour_preview: None,
            colour_pair_pending: false,
            full_raster: false,
            raster_chroma: crate::preview::RasterChroma::Active,
            teletext_headers: false,
            colour_width: crate::preview::WIDTH,
            vbi: None,
            dropouts: false,
            dropout_threshold: 0.35,
            dropout_fields: 0,
        })
    }
    /// Pass teletext header images from the VBI receiver to the preview and
    /// to the MKV subtitle track.
    fn forward_headers(&mut self) -> Result<(), String> {
        let Some(server) = &mut self.vbi else {
            return Ok(());
        };
        for header in server.take_headers() {
            if !self.teletext_headers {
                self.teletext_headers = true;
                println!(
                    "TELETEXT_HEADER receiving=true image={}x{} field={}",
                    header.width, header.height, header.field
                );
            }
            self.send_preview(&crate::preview::header_packet(
                header.width,
                header.height,
                header.field,
                &header.rgba,
            ))?;
            if let Some(direct) = &mut self.direct {
                direct.header(header)?;
            }
        }
        Ok(())
    }
    /// `planes` holds the luma plane followed by the chroma plane.
    fn luma_field(&self, planes: &[u16], field: u64, first: bool) -> crate::preview::PreviewFrame {
        if self.full_raster {
            let composite = self.raster_chroma == crate::preview::RasterChroma::Composite;
            preview_field_full(planes, composite, field, first)
        } else {
            preview_field(&planes[..WIDTH * HEIGHT], field, first)
        }
    }
    /// Send one preview packet without waiting; the GUI drops stale packets.
    /// A broken preview ends preview-only mode, but never a recording.
    fn send_preview(&mut self, packet: &[u8]) -> Result<(), String> {
        let Some(socket) = &mut self.socket else {
            return Ok(());
        };
        match socket.send(packet) {
            Ok(_) => Ok(()),
            Err(e) if self.monitor => Err(format!("Live preview connection: {e}")),
            Err(e) => {
                eprintln!("Live preview disabled, recording continues: {e}");
                self.socket = None;
                Ok(())
            }
        }
    }
    fn publish_colour(&mut self, pixels: &[u8], seq: u64, first: bool) -> Result<(), String> {
        let height = if self.full_raster {
            crate::preview::FULL_HEIGHT
        } else {
            crate::preview::HEIGHT
        };
        let packet =
            crate::preview::colour_field_packet(self.colour_width, height, seq, first, pixels);
        if self.socket.is_some() {
            self.send_preview(&packet)?;
        } else if let Some(frame) = self
            .monitor_assembler
            .push(crate::preview::PreviewFrame::parse(&packet)?)
        {
            if self.monitor
                || self
                    .preview_updated
                    .is_none_or(|t| t.elapsed().as_millis() >= 200)
            {
                if let Some(path) = &self.preview {
                    write_preview_packet(path, &crate::preview::frame_packet(&frame))?;
                }
                self.preview_updated = Some(std::time::Instant::now());
            }
            self.last_colour_preview = Some(frame);
        }
        Ok(())
    }
    fn monitor_progress(&mut self, seq: usize, first: bool, detected: usize, recovered: bool) {
        self.count = seq;
        self.last_first = Some(first);
        if self.checkpoint.elapsed().as_secs_f32() >= 1.0 {
            println!("LIVE_PREVIEW fields={seq} sync={}% estimated_clock={recovered} colour={} recording=false", detected * 100 / (HEIGHT + 1), self.colour_preview);
            self.checkpoint = std::time::Instant::now();
        }
    }
    fn metadata(&mut self) -> Result<(), String> {
        if self.monitor || self.direct.is_some() {
            return Ok(());
        }
        self.y
            .as_mut()
            .unwrap()
            .flush()
            .map_err(|e| e.to_string())?;
        self.c
            .as_mut()
            .unwrap()
            .flush()
            .map_err(|e| e.to_string())?;
        let data = json!({"videoParameters":{"system":"PAL","tapeFormat":"VHS","fieldWidth":WIDTH,"fieldHeight":HEIGHT,
            "sampleRate":OUT_RATE,"black16bIre":crate::preview::BLACK,"blanking16bIre":16384,"white16bIre":crate::preview::WHITE,
            "colourBurstStart":98,"colourBurstEnd":138,"activeVideoStart":185,"activeVideoEnd":1107,
            "numberOfSequentialFields":self.fields.len(),"version":format!("oxide-rf native experimental {}",env!("CARGO_PKG_VERSION")),
            "isMapped":false},"fields":self.fields,"nativeDecoder":{"experimental":true,
            "lumaLevelMode":if self.adaptive_black {"automatic"} else {"reference"},"levelAdjust":0.1,"chromaPeakAmplitude":5000,"chromaDelaySamples":2,
            "dropoutCorrection":self.dropouts,"dropoutThreshold":self.dropout_threshold,
            "limitations":["Validated only on a short PAL capture, not a complete tape","Basic chroma burst correction","Spatial dropout concealment cannot recover lost picture detail","No built-in teletext packet decoding"]}});
        let path = PathBuf::from(format!("{}.tbc.json", self.base.display()));
        let temp = path.with_extension("json.tmp");
        fs::write(&temp, serde_json::to_vec(&data).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        fs::rename(temp, path).map_err(|e| e.to_string())
    }
    fn field(
        &mut self,
        data: &Samples,
        pulses: &PulseFinder,
        bounds: (u64, u64),
        fs: f64,
        source_offset: u64,
        mut gpu: Option<&mut crate::gpu::Pipeline>,
    ) -> Result<(), String> {
        let (start, end) = bounds;
        let line = 64e-6 * fs;
        let anchor = pulses
            .horizontal
            .iter()
            .find(|&&p| p >= start + (480e-6 * fs) as u64 && p < end)
            .copied();
        let (mut origin, mut first) = if let Some(anchor) = anchor {
            pal_field_origin(anchor as f64, start as f64, line)
        } else {
            let first = self.last_first.is_none_or(|v| !v);
            (
                start as f64 + if first { 2.5 * line } else { 3.0 * line },
                first,
            )
        };
        if let Some(previous) = self.last_first {
            let expected = !previous;
            if first != expected {
                first = expected;
                let full = start as f64 + if first { 2.5 * line } else { 3.0 * line };
                origin = anchor.map_or(full, |p| {
                    p as f64 - ((p as f64 - full) / line).round() * line
                });
            }
        }
        let mut locations = Vec::new();
        let mut locked = Vec::new();
        let mut porches = Vec::new();
        let mut tips = Vec::new();
        for row in 0..=HEIGHT {
            let predicted = origin + row as f64 * line;
            let index = pulses
                .horizontal
                .partition_point(|&p| (p as f64) < predicted);
            let best = [index.checked_sub(1), Some(index)]
                .into_iter()
                .flatten()
                .filter_map(|i| pulses.horizontal.get(i))
                .filter(|&&p| (p as f64 - predicted).abs() < line * 0.15)
                .min_by(|a, b| {
                    (**a as f64 - predicted)
                        .abs()
                        .total_cmp(&(**b as f64 - predicted).abs())
                });
            locked.push(best.is_some());
            let position = best.map_or(predicted, |&p| {
                if gpu.is_some() {
                    return p as f64;
                }
                let relative = p as f64 - data.base as f64;
                let refined = refine_hsync_at_level(&data.sync, relative, fs, None)
                    .and_then(|p| refine_hsync_at_level(&data.sync, p, fs, None));
                let Some(refined) = refined else {
                    *locked.last_mut().unwrap() = false;
                    return p as f64;
                };
                if (22..300).contains(&row) {
                    let (tip, porch) = sync_levels(&data.sync, refined, fs);
                    tips.push(tip);
                    porches.push(porch);
                }
                data.base as f64 + refined
            });
            locations.push(position);
        }
        if let Some(pipeline) = gpu.as_deref_mut() {
            let levels = pipeline.refine(&mut locations, &mut locked)?;
            for row in 22..300 {
                if locked[row] {
                    tips.push(levels[row].0);
                    porches.push(levels[row].1);
                }
            }
        }
        if tips.len() > 100 && porches.len() > 100 {
            let threshold = (median(&mut tips) + median(&mut porches)) * 0.5;
            if let Some(pipeline) = gpu.as_deref_mut() {
                pipeline.refine_at_level(&mut locations, &mut locked, threshold)?;
            } else {
                for (position, valid) in locations.iter_mut().zip(&mut locked) {
                    if *valid {
                        if let Some(refined) = refine_hsync_at_level(
                            &data.sync,
                            *position - data.base as f64,
                            fs,
                            Some(threshold),
                        )
                        .and_then(|p| refine_hsync_at_level(&data.sync, p, fs, Some(threshold)))
                        {
                            *position = data.base as f64 + refined;
                        } else {
                            *valid = false;
                        }
                    }
                }
            }
        }
        reject_sync_outliers(&locations, &mut locked, fs);
        let detected = locked.iter().filter(|v| **v).count();
        repair_line_locations(&mut locations, &locked, line);
        let blank = if porches.len() > 100 {
            median(&mut porches)
        } else {
            4.1e6
        };
        let measured_blank = blank;
        let blank = if self.adaptive_black && (3.95e6..4.5e6).contains(&blank) {
            blank
        } else {
            4.1e6
        };
        if std::env::var_os("VHS_NATIVE_STATS").is_some() {
            let tip = if tips.is_empty() {
                0.0
            } else {
                median(&mut tips)
            };
            eprintln!("FIELD_LEVELS tip_hz={tip:.0} blank_hz={blank:.0} measured_blank_hz={measured_blank:.0}");
        }
        if locations[0] < (data.base + 7) as f64
            || locations[HEIGHT] > (data.base + data.len() as u64 - 8) as f64
        {
            return Err("Not enough samples for a complete field".into());
        }
        let seq = self.count + 1;
        let dropout_map = if self.dropouts {
            let map = if let Some(pipeline) = gpu.as_deref_mut() {
                let map = pipeline.dropout_map(&locations, self.dropout_threshold)?;
                pipeline.set_dropout_mask(&map)?;
                map
            } else {
                let start = locations[0].floor() as u64;
                let begin = (start - data.base) as usize;
                let end = (locations[HEIGHT].ceil() as u64 - data.base) as usize;
                crate::dropout::detect(
                    &data.envelope[begin..end],
                    start,
                    &locations,
                    fs,
                    self.dropout_threshold,
                )
            };
            if map.count() > 0 {
                self.dropout_fields += 1;
                println!(
                    "DROPOUTS field={seq} spans={} correction_enabled=true",
                    map.count()
                );
            }
            Some(map)
        } else {
            None
        };
        if self.monitor {
            if let Some(server) = &mut self.vbi {
                let samples = if let Some(pipeline) = gpu.as_deref_mut() {
                    pipeline.vbi(&locations, &locked, blank)?
                } else {
                    let mut samples = Vec::with_capacity(crate::vbi::WIDTH * crate::vbi::LINES);
                    for row in crate::vbi::FIRST_ROW..crate::vbi::FIRST_ROW + crate::vbi::LINES {
                        for x in 0..WIDTH {
                            let hz = interpolate_video(
                                &data.luma,
                                locations[row] - data.base as f64
                                    + x as f64 * (locations[row + 1] - locations[row])
                                        / WIDTH as f64,
                            );
                            samples.push(
                                (16384.0 + (hz - blank) * (376.32 / 7000.0))
                                    .round()
                                    .clamp(0.0, 65535.0) as u16,
                            );
                        }
                    }
                    samples
                };
                server.publish(&samples, first, seq as u64)?;
            }
            self.forward_headers()?;
        }
        if self.monitor && !self.colour_preview && !self.dropouts && !self.full_raster {
            let pixels = if let Some(pipeline) = gpu {
                pipeline.preview(&locations, &locked, blank)?
            } else {
                let mut pixels = Vec::with_capacity(crate::preview::WIDTH * crate::preview::HEIGHT);
                for row in 0..crate::preview::HEIGHT {
                    let row = row + 22;
                    for x in 0..crate::preview::WIDTH {
                        let src =
                            185.0 + (x as f64 + 0.5) * 922.0 / crate::preview::WIDTH as f64 - 0.5;
                        let hz = interpolate_video(
                            &data.luma,
                            locations[row] - data.base as f64
                                + src * (locations[row + 1] - locations[row]) / WIDTH as f64,
                        );
                        pixels.push(crate::preview::gray(
                            16384.0 + (hz - blank) * (376.32 / 7000.0),
                        ));
                    }
                }
                pixels
            };
            if self.socket.is_some() {
                self.send_preview(&crate::preview::field_packet(seq as u64, first, &pixels))?;
            } else if let Some(path) = &self.preview {
                let field = crate::preview::PreviewFrame {
                    width: crate::preview::WIDTH,
                    height: crate::preview::HEIGHT,
                    field: seq as u64,
                    pixels,
                    first_field: Some(first),
                    colour: false,
                };
                if let Some(frame) = self.monitor_assembler.push(field) {
                    preview_pixels_file(
                        path,
                        &frame.pixels,
                        frame.field,
                        frame.width,
                        frame.height,
                    )?;
                }
            }
            self.monitor_progress(seq, first, detected, pulses.recovered.contains(&start));
            return Ok(());
        }
        if self.monitor && self.colour_preview && !self.dropouts {
            if let Some(pipeline) = gpu.as_deref_mut() {
                let pixels = pipeline.preview_colour(&locations, &locked, blank)?;
                self.publish_colour(&pixels, seq as u64, first)?;
                self.monitor_progress(seq, first, detected, pulses.recovered.contains(&start));
                return Ok(());
            }
        }
        let mut pixels = if let Some(pipeline) = gpu.as_deref_mut() {
            pipeline.field(&locations, &locked, blank)?
        } else {
            let mut y = Vec::with_capacity(WIDTH * HEIGHT);
            let mut c = Vec::with_capacity(WIDTH * HEIGHT);
            let mut bursts: Vec<(f64, f32)> = (0..HEIGHT)
                .map(|row| {
                    let a = locations[row] - data.base as f64;
                    let step = (locations[row + 1] - locations[row]) / WIDTH as f64;
                    let burst: Vec<C> = (98..138)
                        .map(|x| interpolate_c(&data.chroma, a + x as f64 * step))
                        .collect();
                    let sum = burst.iter().fold(C::default(), |s, v| C {
                        re: s.re + v.re,
                        im: s.im + v.im,
                    });
                    let phase = sum.im.atan2(sum.re) as f64;
                    let amplitude = (sum.re.hypot(sum.im) / burst.len() as f32).max(1e-8);
                    let gain = if locked[row] { 5000.0 / amplitude } else { 0.0 };
                    (phase, gain)
                })
                .collect();
            smooth_chroma_gains(&mut bursts);
            let flip = pal_chroma_flip(&bursts);
            for row in 0..HEIGHT {
                let a = locations[row] - data.base as f64;
                let step = (locations[row + 1] - locations[row]) / WIDTH as f64;
                let (phase, gain) = bursts[row];
                let target = pal_burst_phase(row)
                    + if flip {
                        if row % 2 == 0 {
                            std::f64::consts::FRAC_PI_2
                        } else {
                            -std::f64::consts::FRAC_PI_2
                        }
                    } else {
                        0.0
                    };
                for x in 0..WIDTH {
                    let p = a + x as f64 * step;
                    let hz = interpolate_video(&data.luma, p);
                    let code = 16384.0 + (hz - blank) * (376.32 / 7000.0);
                    y.push(code.round().clamp(0.0, 65535.0) as u16);
                    // Align the zero-phase colour-under path with the luma
                    // de-emphasis path (two output samples in the PAL profile).
                    let under = interpolate_c(&data.chroma, p - 2.0 * step);
                    // VHS uses the difference sideband of the fSC + colour-under
                    // heterodyne, which conjugates the analytic colour-under signal.
                    let baseband = restore_chroma(under, phase, target);
                    let carrier = C::polar(std::f64::consts::TAU * FSC * x as f64 / OUT_RATE);
                    c.push(
                        (32767.0 + baseband.mul(carrier).re * gain)
                            .round()
                            .clamp(0.0, 65535.0) as u16,
                    );
                }
            }
            y.extend(c);
            y
        };
        if gpu.is_none() {
            if let Some(map) = &dropout_map {
                crate::dropout::repair(&mut pixels, map);
            }
        }
        // While recording, the preview file is refreshed every 200 ms. Render only
        // the adjacent field pair needed for that frame instead of every field.
        let colour_due = self.monitor
            || self.socket.is_some()
            || self.colour_pair_pending
            || self
                .preview_updated
                .is_none_or(|t| t.elapsed().as_millis() >= 200);
        if self.colour_preview && colour_due {
            let updated = self.preview_updated;
            let colour = if let Some(pipeline) = gpu {
                pipeline.colour_pixels()?
            } else {
                let cpu = self
                    .colour_cpu
                    .as_mut()
                    .ok_or("CPU colour preview is not configured")?;
                if self.full_raster {
                    cpu.preview_full(&pixels, self.colour_width, self.raster_chroma)
                } else {
                    cpu.preview(&pixels)
                }
            };
            self.publish_colour(&colour, seq as u64, first)?;
            if self.monitor {
                self.monitor_progress(seq, first, detected, pulses.recovered.contains(&start));
                return Ok(());
            }
            self.colour_pair_pending = self.preview_updated == updated;
        }
        let (y, c) = pixels.split_at(WIDTH * HEIGHT);
        if self.monitor {
            let field = self.luma_field(&pixels, seq as u64, first);
            if self.socket.is_some() {
                self.send_preview(&crate::preview::sized_field_packet(
                    field.width,
                    field.height,
                    seq as u64,
                    first,
                    &field.pixels,
                ))?;
            } else if let Some(path) = &self.preview {
                if let Some(frame) = self.monitor_assembler.push(field) {
                    preview_pixels_file(
                        path,
                        &frame.pixels,
                        frame.field,
                        frame.width,
                        frame.height,
                    )?;
                }
            }
            self.monitor_progress(seq, first, detected, pulses.recovered.contains(&start));
            return Ok(());
        }
        if let Some(server) = &mut self.vbi {
            server.publish(
                &y[crate::vbi::FIRST_ROW * WIDTH
                    ..(crate::vbi::FIRST_ROW + crate::vbi::LINES) * WIDTH],
                first,
                seq as u64,
            )?;
        }
        self.forward_headers()?;
        if let Some(direct) = &mut self.direct {
            direct.field(&pixels, first, (source_offset + start) as f64, seq as u64)?;
        } else {
            write_plane(self.y.as_mut().unwrap(), y)?;
            write_plane(self.c.as_mut().unwrap(), c)?;
        }
        self.count = seq;
        self.last_first = Some(first);
        if self.direct.is_none() {
            self.fields.push(json!({"seqNo":seq,"isFirstField":first,"syncConf":detected*100/(HEIGHT+1),
            "fieldPhaseID":((seq-1)%8)+1,"diskLoc":(source_offset+start) as f64/(fs*0.02),
            "fileLoc":source_offset+start,"decodeFaults":(if detected<290 {1} else {0}) | (if pulses.recovered.contains(&start){4}else{0}),"burstStartLine":if detected<10 {-1}else{0},"isEstimatedClock":pulses.recovered.contains(&start),"isDuplicateField":false}));
            if let Some(map) = dropout_map.filter(|m| m.count() > 0) {
                self.fields.last_mut().unwrap()["dropOuts"] = map.json();
            }
        }
        if self.checkpoint.elapsed().as_secs_f32() >= 2.0 {
            self.metadata()?;
            self.checkpoint = std::time::Instant::now();
        }
        if self.socket.is_some() && !self.colour_preview {
            // Real-time luma preview while recording, one packet per field.
            let field = self.luma_field(&pixels, seq as u64, first);
            self.send_preview(&crate::preview::sized_field_packet(
                field.width,
                field.height,
                seq as u64,
                first,
                &field.pixels,
            ))?;
        } else if let Some(path) = self.preview.as_ref().filter(|_| !self.colour_preview) {
            if self
                .preview_updated
                .is_none_or(|t| t.elapsed().as_millis() >= 200)
            {
                self.preview_updated = Some(std::time::Instant::now());
                if let Err(e) = preview_file(
                    path,
                    y,
                    seq as u64,
                    first,
                    self.last_preview.as_ref().map(|p| p.0.as_slice()),
                ) {
                    eprintln!("Preview: {e}");
                }
            }
            pixels.truncate(WIDTH * HEIGHT);
            self.previous_preview = self.last_preview.take();
            self.last_preview = Some((pixels, seq as u64));
        }
        println!(
            "Frame {} · field {} · sync {}%{}",
            seq.div_ceil(2),
            seq,
            detected * 100 / (HEIGHT + 1),
            if pulses.recovered.contains(&start) {
                " · estimated field clock"
            } else {
                ""
            }
        );
        Ok(())
    }
}
fn write_plane(writer: &mut impl Write, pixels: &[u16]) -> Result<(), String> {
    #[cfg(target_endian = "little")]
    let bytes = unsafe {
        // A shared u16 slice can be viewed as initialized bytes for its exact lifetime.
        std::slice::from_raw_parts(pixels.as_ptr().cast::<u8>(), std::mem::size_of_val(pixels))
    };
    #[cfg(target_endian = "big")]
    let storage: Vec<u8> = pixels.iter().flat_map(|x| x.to_le_bytes()).collect();
    #[cfg(target_endian = "big")]
    let bytes = storage.as_slice();
    writer.write_all(bytes).map_err(|e| e.to_string())
}

/// Every TBC sample and line of a field as 8-bit luma, padded to the even
/// full-raster preview size; sync and blanking below black are clipped. With
/// `composite`, the raw chroma signal is added, showing burst and subcarrier.
fn preview_field_full(
    planes: &[u16],
    composite: bool,
    field: u64,
    first: bool,
) -> crate::preview::PreviewFrame {
    let (width, height) = (crate::preview::FULL_WIDTH, crate::preview::FULL_HEIGHT);
    let mut pixels = vec![crate::preview::gray(crate::preview::BLACK); width * height];
    let plane = WIDTH * HEIGHT;
    for row in 0..HEIGHT {
        for x in 0..WIDTH {
            let at = row * WIDTH + x;
            let mut level = planes[at] as f32;
            if composite {
                level += planes[plane + at] as f32 - 32767.0;
            }
            pixels[row * width + x] = crate::preview::gray(level);
        }
    }
    crate::preview::PreviewFrame {
        width,
        height,
        field,
        pixels,
        first_field: Some(first),
        colour: false,
    }
}
fn preview_field(y: &[u16], field: u64, first: bool) -> crate::preview::PreviewFrame {
    let mut pixels = Vec::with_capacity(crate::preview::WIDTH * crate::preview::HEIGHT);
    for row in 0..crate::preview::HEIGHT {
        let row = row + 22;
        for x in 0..crate::preview::WIDTH {
            let sx = 185.0 + (x as f32 + 0.5) * 922.0 / crate::preview::WIDTH as f32 - 0.5;
            let i = sx as usize;
            let tx = sx - i as f32;
            let sample = |r: usize| {
                cubic(
                    y[r * WIDTH + i - 1] as f32,
                    y[r * WIDTH + i] as f32,
                    y[r * WIDTH + i + 1] as f32,
                    y[r * WIDTH + i + 2] as f32,
                    tx,
                )
            };
            pixels.push(crate::preview::gray(sample(row)));
        }
    }
    crate::preview::PreviewFrame {
        width: crate::preview::WIDTH,
        height: crate::preview::HEIGHT,
        field,
        pixels,
        first_field: Some(first),
        colour: false,
    }
}
fn preview_file(
    path: &Path,
    y: &[u16],
    field: u64,
    first: bool,
    previous: Option<&[u16]>,
) -> Result<(), String> {
    let Some(previous) = previous else {
        return Ok(());
    };
    let mut assembler = crate::preview::FrameAssembler::default();
    assembler.push(preview_field(previous, field - 1, !first));
    if let Some(frame) = assembler.push(preview_field(y, field, first)) {
        preview_pixels_file(path, &frame.pixels, field, frame.width, frame.height)?;
    }
    Ok(())
}
fn preview_pixels_file(
    path: &Path,
    pixels: &[u8],
    field: u64,
    width: usize,
    height: usize,
) -> Result<(), String> {
    let packet = crate::preview::packet(width as u32, height as u32, field, pixels);
    write_preview_packet(path, &packet)
}
fn write_preview_packet(path: &Path, packet: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let temp = path.with_extension("tmp");
    fs::write(&temp, packet).map_err(|e| e.to_string())?;
    fs::rename(temp, path).map_err(|e| e.to_string())
}

pub fn decode(args: &[String]) -> Result<(), String> {
    let allowed = [
        "--frequency",
        "--input",
        "--output",
        "--start",
        "--frames",
        "--preview",
        "--backend",
        "--device",
        "--sample-format",
        "--input-mode",
        "--preview-only",
        "--preview-port",
        "--luma-levels",
        "--preview-colour",
        "--vbi-bind",
        "--vbi-wait",
        "--dropout-correct",
        "--dropout-threshold",
        "--audio-monitor",
        "--record-mode",
        "--mkv-profile",
        "--mkv-encoder",
        "--audio-offset-ms",
        "--full-raster",
        "--full-raster-chroma",
        "--burn-teletext-header",
        "--teletext-header-x",
        "--teletext-header-y",
        "--teletext-header-size",
    ];
    if !args.len().is_multiple_of(2)
        || args
            .as_chunks::<2>()
            .0
            .iter()
            .any(|p| !allowed.contains(&p[0].as_str()))
    {
        return Err("Invalid arguments: expected --option value pairs".into());
    }
    let get = |key: &str, default: &str| {
        args.windows(2)
            .find(|p| p[0] == key)
            .map(|p| p[1].clone())
            .unwrap_or(default.into())
    };
    let mut fs = get("--frequency", "28.636363636363637")
        .parse::<f64>()
        .map_err(|e| e.to_string())?
        * 1e6;
    if !(14e6..=100e6).contains(&fs) {
        return Err("Supported sample rate: 14–100 MHz".into());
    }
    let input = PathBuf::from(get("--input", ""));
    let monitor = match get("--preview-only", "false").as_str() {
        "true" => true,
        "false" => false,
        _ => return Err("Preview-only must be true or false".into()),
    };
    let direct_mkv = match get("--record-mode", "tbc").as_str() {
        "tbc" => false,
        "mkv" if !monitor => true,
        "mkv" => return Err("Direct MKV requires recording; disable preview-only".into()),
        _ => return Err("Record mode must be tbc or mkv".into()),
    };
    let audio_monitor = match get("--audio-monitor", "false").as_str() {
        "true" => true,
        "false" => false,
        _ => return Err("Audio monitor must be true or false".into()),
    };
    let base = PathBuf::from(get("--output", ""));
    let colour_preview = match get("--preview-colour", "false").as_str() {
        "true" => true,
        "false" => false,
        _ => return Err("Preview colour must be true or false".into()),
    };
    let full_raster = match get("--full-raster", "false").as_str() {
        "true" => true,
        "false" => false,
        _ => return Err("Full raster must be true or false".into()),
    };
    let raster_chroma = crate::preview::RasterChroma::parse(&get("--full-raster-chroma", "active"))
        .ok_or("Full-raster chroma must be active, whole or composite")?;
    // A composite raster is grey: the chroma signal is part of the picture.
    let colour_preview = colour_preview
        && !(full_raster && raster_chroma == crate::preview::RasterChroma::Composite);
    let dropouts = match get("--dropout-correct", "false").as_str() {
        "true" => true,
        "false" => false,
        _ => return Err("Dropout correction must be true or false".into()),
    };
    let dropout_threshold = match get("--dropout-threshold", "").as_str() {
        "" => crate::dropout::default_threshold(fs / 1e6),
        value => value
            .parse::<f32>()
            .map_err(|_| "Invalid dropout threshold")?,
    };
    if !(0.05..=0.8).contains(&dropout_threshold) {
        return Err("Dropout threshold must be 0.05–0.8".into());
    }
    if !monitor && base.as_os_str().is_empty() {
        return Err("Missing output base name".into());
    }
    let start = get("--start", "0")
        .parse::<u64>()
        .map_err(|e| e.to_string())?;
    let frames = get("--frames", "0")
        .parse::<usize>()
        .map_err(|e| e.to_string())?;
    let preview = args
        .windows(2)
        .find(|p| p[0] == "--preview")
        .map(|p| PathBuf::from(&p[1]));
    // The GUI listens on a loopback port for live preview packets.
    let socket = match args.windows(2).find(|p| p[0] == "--preview-port") {
        Some(port) => Some(
            crate::channel::PreviewSender::connect(
                port[1].parse().map_err(|_| "Invalid preview port")?,
            )
            .map_err(|e| format!("Preview connection: {e}"))?,
        ),
        None => None,
    };
    if monitor && socket.is_none() && preview.is_none() {
        return Err("Preview-only requires --preview-port or --preview".into());
    }
    let backend = get("--backend", "opencl");
    if !["cpu", "opencl"].contains(&backend.as_str()) {
        return Err("Backend must be cpu or opencl".into());
    }
    let input_mode = get("--input-mode", "auto");
    let network = input_mode == "tcp"
        || (input_mode == "auto" && input.to_string_lossy().starts_with("tcp://"));
    let live = match input_mode.as_str() {
        "tcp" => true,
        "auto" => network || crate::cxadc::is_stream(&input),
        "file" => false,
        "cxadc" => true,
        _ => return Err("Input mode must be auto, file, cxadc or tcp".into()),
    };
    if live && start != 0 {
        return Err("Live RF input requires start frame 0".into());
    }
    let mut sample_format = get("--sample-format", "auto");
    if live && !network {
        let info = crate::cxadc::inspect(&input);
        if sample_format == "auto" {
            sample_format = info
                .sample_format()
                .ok_or("Cannot detect live sample format; select u8 or u16 explicitly")?
                .into();
        }
        if let Some(actual) = info.sample_format() {
            if sample_format != actual {
                return Err(format!("cxadc tenbit setting requires {actual}, selected {sample_format}. Refresh device settings."));
            }
        }
        println!("LIVE_DEVICE {} · {}", input.display(), info.summary());
    }
    let gpu = backend == "opencl";
    std::env::set_var("VHS_GPU_DEVICE", get("--device", "0"));
    install_stop_handler()?;
    let mut vbi = match get("--vbi-bind", "").as_str() {
        "" => None,
        address => Some(crate::vbi::Server::bind(address)?),
    };
    match get("--vbi-wait", "false").as_str() {
        "true" => {
            if let Some(server) = &mut vbi {
                server.wait_for_client()?;
            }
        }
        "false" => {}
        _ => return Err("VBI wait must be true or false".into()),
    }
    if stop_requested() {
        return Ok(());
    }
    let connection = if network {
        let connection = crate::network::Connection::connect(&input.to_string_lossy())?;
        let format = if connection.profile.bytes == 1 {
            "u8"
        } else {
            "u16"
        };
        if sample_format != "auto" && sample_format != format {
            return Err(format!("RF sender uses {format}, selected {sample_format}"));
        }
        sample_format = format.into();
        fs = connection.profile.rate;
        Some(connection)
    } else {
        None
    };
    let started = std::time::Instant::now();
    let fft = Fft::new(false);
    let mut pipeline = if gpu {
        Some(if monitor && !colour_preview && !full_raster {
            crate::gpu::Pipeline::new_monitor(fs)?
        } else if monitor {
            crate::gpu::Pipeline::new_colour_monitor(fs)?
        } else {
            crate::gpu::Pipeline::new(fs)?
        })
    } else {
        None
    };
    let colour_width = if full_raster {
        crate::preview::FULL_WIDTH
    } else {
        crate::preview::WIDTH
    };
    if colour_preview {
        if let Some(pipeline) = &mut pipeline {
            pipeline.enable_colour_preview(full_raster.then_some((colour_width, raster_chroma)))?;
        }
    }
    if dropouts {
        if let Some(pipeline) = &mut pipeline {
            pipeline.enable_dropouts()?;
        }
    }
    let initialized = started.elapsed();
    let mut demod_time = std::time::Duration::ZERO;
    let mut field_time = std::time::Duration::ZERO;
    let adaptive_black = match get("--luma-levels", "reference").as_str() {
        "reference" => false,
        "automatic" => true,
        _ => return Err("Luma levels must be reference or automatic".into()),
    };
    // Burning needs header images from a teletext receiver on the VBI stream.
    let defaults = crate::preview::HeaderPlacement::default();
    let number = |key: &str, default: f32| {
        get(key, &default.to_string())
            .parse::<f32>()
            .map_err(|_| format!("Invalid {key}"))
    };
    let placement = crate::preview::HeaderPlacement {
        x: number("--teletext-header-x", defaults.x)?,
        y: number("--teletext-header-y", defaults.y)?,
        size: number("--teletext-header-size", defaults.size)?,
    };
    if !placement.is_valid() {
        return Err("Teletext header position must be 0–1 and size 0.25–3".into());
    }
    let burn_header = match get("--burn-teletext-header", "false").as_str() {
        "true" => vbi.is_some().then_some(placement),
        "false" => None,
        _ => return Err("Burn teletext header must be true or false".into()),
    };
    if let Some(placement) = burn_header.filter(|_| direct_mkv) {
        println!(
            "TELETEXT_HEADER burn_in=true x={} y={} size={} max_delay_frames=100",
            placement.x, placement.y, placement.size
        );
    }
    let direct = if direct_mkv {
        Some(crate::export::DirectMkvWorker::new(
            &base,
            &get("--mkv-profile", "archive"),
            &get("--mkv-encoder", "auto"),
            gpu,
            get("--device", "0")
                .parse()
                .map_err(|_| "Invalid OpenCL device index")?,
            get("--audio-offset-ms", "0")
                .parse()
                .map_err(|_| "Invalid audio offset")?,
            full_raster.then_some(raster_chroma),
            burn_header,
        )?)
    } else {
        None
    };
    let mut output = Output::open(&base, preview, monitor, socket, direct)?;
    output.adaptive_black = adaptive_black;
    output.vbi = vbi;
    output.dropouts = dropouts;
    output.dropout_threshold = dropout_threshold;
    output.colour_preview = colour_preview;
    output.full_raster = full_raster;
    output.raster_chroma = raster_chroma;
    output.colour_width = colour_width;
    if colour_preview && !gpu {
        output.colour_cpu = Some(crate::export::Cpu::new());
    }
    let mut network_audio = None;
    let mut source = if let Some(connection) = connection {
        network_audio = connection.audio(
            if monitor { None } else { Some(base.as_path()) },
            audio_monitor,
        )?;
        Source::Live(connection.start()?)
    } else if live {
        Source::Live(crate::cxadc::Stream::open(&input, &sample_format, fs)?)
    } else {
        Source::open(&input, &sample_format)?
    };
    let source_offset = (start as f64 * 0.04 * fs).round() as u64;
    source.skip(source_offset)?;
    let mut live_stats = match &source {
        Source::Live(stream) if !monitor => Some(crate::diagnostics::LiveStats::new(
            &base,
            fs * stream.bytes_per_sample() as f64,
            output
                .direct
                .as_ref()
                .map_or(0, |direct| direct.encoder_pid()),
            &format!(
                "opencl_device={:?} rf_mhz={:.6} args={args:?}",
                if gpu {
                    get("--device", "0")
                        .parse::<usize>()
                        .ok()
                        .and_then(|i| crate::devices().into_iter().nth(i))
                        .unwrap_or_default()
                } else {
                    "cpu".into()
                },
                fs / 1e6
            ),
        )),
        _ => None,
    };
    let mut counters = LiveCounters::default();
    println!("Native Rust PAL/VHS decoder · EXPERIMENTAL · no Python");
    println!(
        "Backend: {} · {:.9} MHz",
        if gpu && monitor && colour_preview {
            "OpenCL live PAL colour preview · no recording"
        } else if gpu && monitor {
            "OpenCL luma-only live preview · no recording"
        } else if gpu {
            "OpenCL resident RF + sync + TBC + chroma"
        } else {
            "Rust CPU"
        },
        fs / 1e6
    );
    let batch_samples = if gpu {
        pipeline.as_ref().unwrap().batch_samples()
    } else {
        N
    };
    let mut raw = Vec::new();
    let mut absolute = 0;
    let mut data = Samples {
        luma: Vec::new(),
        envelope: Vec::new(),
        sync: Vec::new(),
        chroma: Vec::new(),
        base: EDGE as u64,
        resident_len: if gpu { Some(0) } else { None },
    };
    let mut pulses = PulseFinder::default();
    let mut processed = 0usize;
    let mut rate_report = (std::time::Instant::now(), 0usize, std::time::Duration::ZERO);
    let result = (|| -> Result<(), String> {
        raw = source.read(batch_samples)?;
        while raw.len() >= N && !STOP.load(Ordering::Relaxed) {
            if raw.iter().any(|x| !x.is_finite()) {
                return Err("RF contains NaN/Inf samples".into());
            }
            let demod_started = std::time::Instant::now();
            let blocks = if gpu { (raw.len() - 2 * EDGE) / HOP } else { 1 };
            let consumed = blocks * HOP;
            if let Some(pipeline) = &mut pipeline {
                if data.len() + consumed > pipeline.capacity() {
                    return Err("OpenCL signal ring would overwrite an undecoded field".into());
                }
                for (start, length) in pipeline.demod(&raw[..consumed + 2 * EDGE], absolute)? {
                    pulses.pulse(start, length, fs);
                }
                *data.resident_len.as_mut().unwrap() += consumed;
            } else {
                let block = if monitor && !colour_preview {
                    demod_mode(&raw, fs, &fft, absolute, true)?
                } else {
                    demod(&raw, fs, &fft, absolute)?
                };
                pulses.push(&block.sync, block.base, fs);
                data.luma.extend(block.luma);
                data.envelope.extend(block.envelope);
                data.sync.extend(block.sync);
                data.chroma.extend(block.chroma);
            }
            pulses.maintain_field_clock(data.base + data.len() as u64, fs);
            demod_time += demod_started.elapsed();
            while pulses.vertical.len() > processed + 1 {
                let a = pulses.vertical[processed];
                let b = pulses.vertical[processed + 1];
                if data.base + (data.len() as u64) < b + (fs * 0.0005) as u64 {
                    break;
                }
                if (0.018..0.022).contains(&((b - a) as f64 / fs)) && a >= data.base {
                    let field_started = std::time::Instant::now();
                    output.field(&data, &pulses, (a, b), fs, source_offset, pipeline.as_mut())?;
                    field_time += field_started.elapsed();
                    if pulses.recovered.contains(&a) {
                        counters.estimated += 1;
                    }
                } else {
                    eprintln!("Skipped incomplete field at {:.3} s", a as f64 / fs);
                    counters.skipped += 1;
                }
                processed += 1;
                if frames > 0 && output.count >= frames * 2 {
                    return Ok(());
                }
                let trim = a.saturating_sub(data.base + (fs * 0.001) as u64) as usize;
                if trim > 0 && trim < data.len() {
                    data.trim(trim);
                }
            }
            if processed > 10 {
                pulses.vertical.drain(..processed);
                processed = 0;
                pulses.horizontal.retain(|&p| p >= data.base);
                pulses.recovered.retain(|&p| p >= data.base);
            }
            // Live recordings report headroom too: below 1.0x the RF buffer fills.
            if (monitor || live) && rate_report.0.elapsed().as_secs_f32() >= 1.0 {
                let fields = output.count - rate_report.1;
                let work = (demod_time + field_time - rate_report.2)
                    .as_secs_f64()
                    .max(0.000001);
                println!(
                    "PREVIEW_RATE frames_per_second={:.2} processing_realtime={:.2} mkv_queue_fields={}",
                    fields as f64 * 0.5 / rate_report.0.elapsed().as_secs_f64(),
                    fields as f64 * 0.02 / work,
                    output.direct.as_ref().map_or(0, |direct| direct.queued())
                );
                rate_report = (
                    std::time::Instant::now(),
                    output.count,
                    demod_time + field_time,
                );
            }
            // Never accumulate an unbounded recording if vertical lock is lost.
            let (limit, retain) = if gpu { (0.075, 0.04) } else { (0.2, 0.1) };
            if data.len() > (fs * limit) as usize {
                let trim = data.len() - (fs * retain) as usize;
                data.trim(trim);
                pulses.horizontal.retain(|&p| p >= data.base);
                pulses.vertical.retain(|&p| p >= data.base);
                processed = 0;
            }
            raw.drain(..consumed);
            let waiting = std::time::Instant::now();
            raw.extend(source.read(batch_samples - raw.len())?);
            counters.input_wait += waiting.elapsed();
            absolute += consumed as u64;
            if let Some(stats) = &mut live_stats {
                stats.update(counters.sample(&source, &output, demod_time, field_time));
            }
        }
        Ok(())
    })();
    if let Some(stats) = &mut live_stats {
        stats.finish(
            counters.sample(&source, &output, demod_time, field_time),
            result.as_ref().err().map(String::as_str),
        );
    }
    // Disconnect acquisition before finalization so audio follows the end of RF
    // and no live buffers grow while the encoder drains or audio is muxed.
    drop(source);
    let audio_result = network_audio
        .as_mut()
        .map(|audio| audio.finish())
        .transpose();
    output.metadata()?;
    let direct_result = output
        .direct
        .as_mut()
        .map(|direct| direct.finish(network_audio.is_some()))
        .transpose();
    if let (Some(path), Some(frame)) = (&output.preview, &output.last_colour_preview) {
        write_preview_packet(path, &crate::preview::frame_packet(frame))?;
    }
    if let (Some(path), Some((pixels, field))) = (&output.preview, &output.last_preview) {
        if let Err(e) = preview_file(
            path,
            pixels,
            *field,
            output.last_first.unwrap_or(true),
            output.previous_preview.as_ref().map(|p| p.0.as_slice()),
        ) {
            eprintln!("Preview: {e}");
        }
    }
    result?;
    audio_result?;
    direct_result?;
    if output.count == 0 && !stop_requested() {
        return Err(
            "No complete PAL field found; check the format, sample rate and RF signal.".into(),
        );
    }
    println!(
        "OUTPUT_OK fields={} bytes_per_plane={} recording={}",
        output.count,
        if monitor || direct_mkv {
            0
        } else {
            output.count * WIDTH * HEIGHT * 2
        },
        !monitor
    );
    let seconds = started.elapsed().as_secs_f64();
    println!("PERFORMANCE wall_s={seconds:.3} init_s={:.3} demod_s={:.3} tbc_write_s={:.3} frames_per_second={:.2} realtime={:.2}x",
        initialized.as_secs_f64(), demod_time.as_secs_f64(), field_time.as_secs_f64(),
        output.count as f64/2.0/seconds, output.count as f64*0.02/seconds);
    if monitor {
        let work = (demod_time + field_time).as_secs_f64().max(0.000001);
        println!(
            "PREVIEW_PERFORMANCE fields={} processing_realtime={:.2}x recording_files=0",
            output.count,
            output.count as f64 * 0.02 / work
        );
    }
    if STOP.load(Ordering::Relaxed) {
        println!(
            "{}",
            if monitor {
                "Preview stopped; no recording was created."
            } else if direct_mkv {
                "Stopped; direct MKV finalized. No TBC files created."
            } else {
                "Stopped; partial outputs and JSON saved."
            }
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tbc_resampling_retains_band_limited_detail() {
        let fs = 315e6 / 11.0;
        let freq = 2.0e6;
        let signal: Vec<f32> = (0..1000)
            .map(|i| (std::f64::consts::TAU * freq * i as f64 / fs).sin() as f32)
            .collect();
        let (mut linear, mut tbc_error) = (0.0, 0.0);
        for i in 10..990 {
            let p = i as f64 + 0.5;
            let expected = (std::f64::consts::TAU * freq * p / fs).sin() as f32;
            linear += (interpolate(&signal, p) - expected).powi(2);
            tbc_error += (interpolate_video(&signal, p) - expected).powi(2);
        }
        assert!(tbc_error < linear * 0.02);
    }
    #[test]
    fn chroma_gain_outliers_do_not_amplify_weak_burst_rows() {
        let mut bursts = vec![(0.0, 1.0); HEIGHT];
        bursts[0].1 = 0.0;
        bursts[150].1 = 1000.0;
        smooth_chroma_gains(&mut bursts);
        assert_eq!(bursts[0].1, 0.0);
        assert!(bursts.iter().skip(1).all(|b| (b.1 - 1.0).abs() < 1e-6));
    }
    #[test]
    fn sync_rejection_preserves_tape_warp_and_head_switch_steps() {
        let fs = 315e6 / 11.0;
        let line = 64e-6 * fs;
        let mut locations: Vec<_> = (0..HEIGHT + 1)
            .map(|row| {
                10000.0
                    + row as f64 * line
                    + 5.0 * (row as f64 / 20.0).sin()
                    + if row >= 300 { 10.0 } else { 0.0 }
            })
            .collect();
        locations[150] += 25.0;
        let mut locked = vec![true; locations.len()];
        reject_sync_outliers(&locations, &mut locked, fs);
        assert!(!locked[150]);
        assert_eq!(locked.iter().filter(|v| !**v).count(), 1);
    }
    #[test]
    fn preview_only_does_not_create_recording_files_or_metadata_directory() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = std::env::temp_dir().join(format!(
            "vhs-preview-no-output-{}-{stamp}",
            std::process::id()
        ));
        let mut output = Output::open(&parent.join("capture"), None, true, None, None).unwrap();
        output.metadata().unwrap();
        assert!(!parent.exists());
        assert!(output.y.is_none() && output.c.is_none() && output.fields.is_empty());
    }
    #[test]
    #[ignore = "requires matching reference/candidate TBC captures"]
    fn real_vbi_rows_and_field_polarity_match_reference() {
        let candidate = std::env::var("VHS_VBI_CANDIDATE").expect("VHS_VBI_CANDIDATE");
        let reference = std::env::var("VHS_VBI_REFERENCE").expect("VHS_VBI_REFERENCE");
        let meta = |p: &str| -> Value {
            serde_json::from_slice(&fs::read(format!("{p}.json")).unwrap()).unwrap()
        };
        let actual_meta = meta(&candidate);
        let reference_meta = meta(&reference);
        let mut a = File::open(candidate).unwrap();
        let mut b = File::open(reference).unwrap();
        let fields = actual_meta["fields"]
            .as_array()
            .unwrap()
            .len()
            .min(reference_meta["fields"].as_array().unwrap().len());
        assert!(fields >= 4, "at least four matching fields are required");
        let mut selected: Vec<_> = (0..4)
            .chain(fields / 2..(fields / 2 + 2).min(fields))
            .chain(fields - 2..fields)
            .collect();
        selected.sort_unstable();
        selected.dedup();
        for field in selected {
            let offset = (field * WIDTH * HEIGHT * 2) as u64;
            a.seek(SeekFrom::Start(offset)).unwrap();
            b.seek(SeekFrom::Start(offset)).unwrap();
            let mut actual = vec![0u8; WIDTH * HEIGHT * 2];
            let mut reference = actual.clone();
            a.read_exact(&mut actual).unwrap();
            b.read_exact(&mut reference).unwrap();
            let read = |v: &[u8], y: usize, x: usize| {
                u16::from_le_bytes([v[(y * WIDTH + x) * 2], v[(y * WIDTH + x) * 2 + 1]]) as f64
            };
            let mut best = (-1.0, 0, 0);
            for dy in -5isize..=5 {
                for dx in -12isize..=12 {
                    let (mut sa, mut sb, mut aa, mut bb, mut ab, mut n) =
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                    for y in 6..22 {
                        for x in (170..1080).step_by(3) {
                            let p = read(&actual, y, x);
                            let q = read(
                                &reference,
                                (y as isize + dy) as usize,
                                (x as isize + dx) as usize,
                            );
                            sa += p;
                            sb += q;
                            aa += p * p;
                            bb += q * q;
                            ab += p * q;
                            n += 1.0;
                        }
                    }
                    let corr =
                        (ab - sa * sb / n) / ((aa - sa * sa / n) * (bb - sb * sb / n)).sqrt();
                    if corr > best.0 {
                        best = (corr, dy, dx);
                    }
                }
            }
            println!(
                "VBI field {field}: corr={:.5}, row_shift={}, sample_shift={}",
                best.0, best.1, best.2
            );
            assert!(best.0 > 0.94, "VBI data do not match the reference field");
            assert_eq!(best.1, 0, "VBI row mapping is shifted");
            assert!(
                best.2.abs() <= 3,
                "horizontal phase lies outside reference tolerance"
            );
            assert_eq!(
                actual_meta["fields"][field]["isFirstField"],
                reference_meta["fields"][field]["isFirstField"]
            );
        }
    }
    #[test]
    fn pal_tbc_starts_at_full_vsync_line_with_correct_field_polarity() {
        let line = 64e-6 * (315e6 / 11.0);
        let start = 1_000_000.0;
        for jitter in [-0.001, 0.0, 0.001] {
            for anchor_line in [7.5, 8.5] {
                let (origin, first) =
                    pal_field_origin(start + (anchor_line + jitter) * line, start, line);
                assert!(first);
                assert!((origin - (start + (2.5 + jitter) * line)).abs() < 1e-6);
            }
            let (origin, first) = pal_field_origin(start + (8.0 + jitter) * line, start, line);
            assert!(!first);
            assert!((origin - (start + (3.0 + jitter) * line)).abs() < 1e-6);
        }
    }
    #[test]
    fn preemphasized_pal_sync_is_not_split_into_short_pulses() {
        let fs = 315e6 / 11.0;
        let mut state = 4.1e6;
        let alpha = 1.0 - (-1.0 / (fs * 0.26e-6_f64)).exp();
        let mut phase = 0.0;
        let raw: Vec<f32> = (0..N)
            .map(|i| {
                let t = i as f64 / fs;
                let time = t % 64e-6;
                let hz = if time < 4.7e-6 {
                    3.8e6
                } else if (10.5e-6..62.5e-6).contains(&time) {
                    4.4e6
                } else {
                    4.1e6
                };
                let emphasized = 5.0 * hz - 4.0 * state;
                state += alpha * (hz - state);
                phase = (phase + std::f64::consts::TAU * emphasized / fs) % std::f64::consts::TAU;
                (phase.cos() + 0.04 * (std::f64::consts::TAU * UNDER * t).cos()) as f32
            })
            .collect();
        let decoded = demod(&raw, fs, &Fft::new(false), 0).unwrap();
        let mut finder = PulseFinder::default();
        finder.push(&decoded.sync, decoded.base, fs);
        assert!(
            finder.horizontal.len() >= 14,
            "found only {} line syncs",
            finder.horizontal.len()
        );
        assert!(finder
            .horizontal
            .windows(2)
            .all(|p| ((p[1] - p[0]) as f64 / fs - 64e-6).abs() < 0.4e-6));
    }

    #[test]
    fn sync_edge_refinement_matches_pal_reference_phase() {
        let fs = 20e6;
        let edge = 300.4;
        for porch in [4.1e6, 4.3e6] {
            let sync: Vec<f32> = (0..1500)
                .map(|i| {
                    let t = (i as f64 - edge) / fs;
                    let mix = if t < 0.5e-6 {
                        ((t + 0.5e-6) / 1e-6).clamp(0.0, 1.0)
                    } else {
                        1.0 - ((t - 4.2e-6) / 1e-6).clamp(0.0, 1.0)
                    };
                    (porch + (3.8e6 - porch) * mix) as f32
                })
                .collect();
            let rough = sync.iter().position(|&v| v < 3.95e6).unwrap() as f64;
            let result = refine_hsync(&sync, rough, fs);
            let expected = edge + (2.25 / 40e6 - 3.0 / OUT_RATE) * fs;
            assert!(
                (result - expected).abs() < 0.05,
                "refined edge {result} vs {expected}"
            );
            let (_, level) = sync_levels(&sync, rough, fs);
            assert!((level as f64 - porch).abs() < 1.0);
        }
    }

    #[test]
    fn shared_sync_threshold_tracks_real_warp_without_following_porch_noise() {
        let fs = 315e6 / 11.0;
        let mut local_error = 0.0;
        for row in 0..32 {
            let edge = 400.2 + 12.0 * (row as f64 * 0.31).sin();
            let porch_noise = 40_000.0 * (row as f64 * 1.73).sin();
            let sync: Vec<f32> = (0..1600)
                .map(|i| {
                    let t = (i as f64 - edge) / fs;
                    let pulse = if t < 0.5e-6 {
                        ((t + 0.5e-6) / 1e-6).clamp(0.0, 1.0)
                    } else {
                        1.0 - ((t - 4.2e-6) / 1e-6).clamp(0.0, 1.0)
                    };
                    (4.1e6 - 300_000.0 * pulse + if t > 6e-6 { porch_noise } else { 0.0 }) as f32
                })
                .collect();
            let expected = edge + (2.25 / 40e6 - 3.0 / OUT_RATE) * fs;
            let measured = refine_hsync_at_level(&sync, edge, fs, Some(3.95e6)).unwrap();
            assert!(
                (measured - expected).abs() < 0.002,
                "row {row}: timing error {}",
                measured - expected
            );
            local_error += (refine_hsync(&sync, edge, fs) - expected).abs();
        }
        assert!(local_error / 32.0 > 0.8);
    }

    #[test]
    fn invalid_sync_pulse_is_not_used_as_a_line_clock() {
        let fs = 20e6;
        let flat = vec![4.1e6; 1500];
        assert!(refine_hsync_at_level(&flat, 300.0, fs, None).is_none());
        let missing_right: Vec<f32> = (0..1500)
            .map(|i| if i >= 300 { 3.8e6 } else { 4.1e6 })
            .collect();
        assert!(refine_hsync_at_level(&missing_right, 300.0, fs, Some(3.95e6)).is_none());
    }

    #[test]
    fn missing_line_syncs_retain_measured_phase_and_clock_at_field_boundaries() {
        let nominal = 64e-6 * 315e6 / 11.0;
        let actual = nominal * 1.0002;
        // A large capture offset must not lose subpixel precision.
        let expected: Vec<_> = (0..314).map(|r| 12e9 + 0.3 + r as f64 * actual).collect();
        let locked: Vec<_> = (0..314)
            .map(|r| (10..309).contains(&r) && !(100..105).contains(&r))
            .collect();
        let mut locations: Vec<_> = expected
            .iter()
            .zip(&locked)
            .enumerate()
            .map(|(r, (&p, &valid))| if valid { p } else { 12e9 + r as f64 * nominal })
            .collect();
        repair_line_locations(&mut locations, &locked, nominal);
        for (row, (&actual, &expected)) in locations.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() < 0.002,
                "row {row}: phase error {}",
                actual - expected
            );
        }
    }
    #[test]
    #[ignore = "requires an explicit real RF capture path"]
    fn real_sync_diagnostic_scan() {
        let path = std::env::var("VHS_RF_PROBE").unwrap();
        let fs = 315e6 / 11.0;
        for second in [0, 1, 2, 3, 4, 5] {
            let mut source = Source::open(Path::new(&path), "auto").unwrap();
            source.skip((second as f64 * fs) as u64).unwrap();
            let mut raw = source.read(N).unwrap();
            let fft = Fft::new(false);
            let mut sync = Vec::new();
            for block in 0..120 {
                let d = demod(&raw, fs, &fft, (block * HOP) as u64).unwrap();
                sync.extend(d.sync);
                raw.drain(..HOP);
                raw.extend(source.read(HOP).unwrap());
            }
            println!("SECOND {second}, adaptive {}", adaptive_sync_slice(&sync));
            for slice in [3.95e6, 4.0e6, 4.05e6, 4.1e6, 4.15e6, 4.2e6] {
                let mut low = None;
                let mut h = 0;
                let mut broad = 0;
                let mut hist = [0u32; 40];
                for (i, &v) in sync.iter().enumerate() {
                    if v < slice {
                        if low.is_none() {
                            low = Some(i);
                        }
                    } else if let Some(start) = low.take() {
                        let us = (i - start) as f64 / fs * 1e6;
                        if (3.0..7.0).contains(&us) {
                            h += 1;
                        }
                        if (20.0..30.5).contains(&us) {
                            broad += 1;
                        }
                        if us < 40.0 {
                            hist[us as usize] += 1;
                        }
                    }
                }
                println!("slice {slice} h={h} broad={broad} widths={hist:?}");
            }
        }
    }
    #[test]
    #[ignore = "requires an explicit real RF capture path"]
    fn real_capture_probe() {
        let path = std::env::var("VHS_RF_PROBE").expect("VHS_RF_PROBE");
        let mut source = Source::open(Path::new(&path), "auto").unwrap();
        let fs = 315e6 / 11.0;
        let fft = Fft::new(false);
        let mut raw = source.read(N).unwrap();
        let mut sync = Vec::new();
        let mut luma = Vec::new();
        let mut pulses = PulseFinder::default();
        for block in 0..120 {
            let decoded = demod(&raw, fs, &fft, (block * HOP) as u64).unwrap();
            pulses.push(&decoded.sync, decoded.base, fs);
            sync.extend(decoded.sync);
            luma.extend(decoded.luma);
            raw.drain(..HOP);
            raw.extend(source.read(HOP).unwrap());
        }
        let quantiles = |v: &[f32]| {
            let mut sorted = v.to_vec();
            sorted.sort_by(f32::total_cmp);
            [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.9, 0.99, 0.999]
                .map(|q| sorted[(q * (sorted.len() - 1) as f64) as usize])
        };
        println!("sync quantiles {:?}", quantiles(&sync));
        println!("luma quantiles {:?}", quantiles(&luma));
        println!(
            "vertical {:?}",
            pulses
                .vertical
                .iter()
                .map(|&v| v as f64 / fs)
                .collect::<Vec<_>>()
        );
        assert!(pulses.vertical.len() >= 6, "missing PAL vertical intervals");
        assert!(
            pulses
                .vertical
                .windows(2)
                .all(|v| ((v[1] - v[0]) as f64 / fs - 0.02).abs() < 10e-6),
            "lost or duplicated fields"
        );
        let mut durations = [0usize; 64];
        let mut low = None;
        for (i, &v) in sync.iter().enumerate() {
            if v < 3.95e6 {
                if low.is_none() {
                    low = Some(i);
                }
            } else if let Some(s) = low.take() {
                let us = ((i - s) as f64 / fs * 1e6).round() as usize;
                if us < durations.len() {
                    durations[us] += 1;
                }
            }
        }
        println!("low pulse widths/us {:?}", durations);
        let mut output = BufWriter::new(File::create("test-results/probe.csv").unwrap());
        writeln!(output, "sample,sync_hz,luma_hz").unwrap();
        for i in (0..sync.len()).step_by(8) {
            writeln!(output, "{},{},{}", i + EDGE, sync[i], luma[i]).unwrap();
        }
        for pair in pulses.vertical.windows(2).take(3) {
            let starts: Vec<_> = pulses
                .horizontal
                .iter()
                .filter(|&&p| p >= pair[0] && p < pair[1])
                .copied()
                .collect();
            assert!(
                starts.len() >= 300,
                "only {} real line syncs in field",
                starts.len()
            );
            println!(
                "field {}..{} hsync={} interval quantiles {:?}",
                pair[0],
                pair[1],
                starts.len(),
                quantiles(
                    &starts
                        .windows(2)
                        .map(|p| (p[1] - p[0]) as f32 / fs as f32 * 1e6)
                        .collect::<Vec<_>>()
                )
            );
        }
    }
    #[test]
    #[ignore = "requires a working OpenCL GPU"]
    fn gpu_demod_matches_cpu_across_ring_wrap_and_large_offsets() {
        for fs in [14e6, 315e6 / 11.0, 100e6] {
            let mut pipeline = crate::gpu::Pipeline::new(fs).unwrap();
            let absolute = (1u64 << 33) + pipeline.capacity() as u64 - (EDGE + HOP / 2) as u64;
            let raw: Vec<_> = (0..N + 2 * HOP)
                .map(|i| {
                    let t = i as f64 / fs;
                    let phase = std::f64::consts::TAU * 4.2e6 * t
                        + 0.2 * (std::f64::consts::TAU * 100e3 * t).sin();
                    (phase.cos() + 0.04 * (std::f64::consts::TAU * UNDER * t).cos()) as f32
                })
                .collect();
            pipeline.demod(&raw, absolute).unwrap();
            let actual = pipeline.read_signal(absolute + EDGE as u64, 3 * HOP);
            for block in 0..3 {
                let expected = demod(
                    &raw[block * HOP..block * HOP + N],
                    fs,
                    &Fft::new(false),
                    absolute + (block * HOP) as u64,
                )
                .unwrap();
                for i in 0..HOP {
                    let p = &actual[(block * HOP + i) * 4..][..4];
                    assert!(
                        (p[0] - expected.luma[i]).abs() < 64.0,
                        "GPU luma mismatch at {fs} Hz, block {block}, sample {i}"
                    );
                    assert!((p[1] - expected.sync[i]).abs() < 64.0, "GPU sync mismatch");
                    assert!(
                        (p[2] - expected.chroma[i].re).abs() < 3e-5,
                        "GPU chroma real mismatch"
                    );
                    assert!(
                        (p[3] - expected.chroma[i].im).abs() < 3e-5,
                        "GPU chroma imaginary mismatch"
                    );
                }
            }
        }
    }
    #[test]
    fn colour_under_difference_sideband_restores_known_pal_colour() {
        for rotation in [0.0, -std::f64::consts::FRAC_PI_2] {
            for row in 0..8 {
                let sign = if row % 2 == 0 { 1.0 } else { -1.0 };
                let desired = C {
                    re: 0.12,
                    im: 0.06 * sign,
                };
                let burst = C {
                    re: -0.3,
                    im: 0.3 * sign,
                };
                let head = C::polar(0.91 + row as f64 * rotation);
                let recorded = C {
                    re: desired.re,
                    im: -desired.im,
                }
                .mul(head);
                let recorded_burst = C {
                    re: burst.re,
                    im: -burst.im,
                }
                .mul(head);
                let phase = recorded_burst.im.atan2(recorded_burst.re) as f64;
                let restored = restore_chroma(recorded, phase, pal_burst_phase(row))
                    .mul(C::polar(row as f64 * std::f64::consts::FRAC_PI_2));
                assert!((restored.re - desired.re).abs() < 1e-6);
                assert!((restored.im - desired.im).abs() < 1e-6);
            }
        }
    }
    #[test]
    fn pal_head_rotation_and_v_switch_are_detected_from_bursts() {
        for rotation in [0.0, -std::f64::consts::FRAC_PI_2] {
            for flip in [false, true] {
                let bursts: Vec<_> = (0..HEIGHT)
                    .map(|row| {
                        let swing = if row % 2 == 0 {
                            135f64.to_radians()
                        } else {
                            225f64.to_radians()
                        };
                        let swing = swing
                            + if flip {
                                if row % 2 == 0 {
                                    std::f64::consts::FRAC_PI_2
                                } else {
                                    -std::f64::consts::FRAC_PI_2
                                }
                            } else {
                                0.0
                            };
                        (
                            0.73 + rotation * row as f64 - swing + row as f64 * 0.001,
                            1.0,
                        )
                    })
                    .collect();
                assert_eq!(pal_chroma_flip(&bursts), flip);
            }
        }
    }
    #[test]
    fn pal_burst_survives_a_two_line_comb_filter() {
        for row in 2..HEIGHT - 2 {
            let a = C::polar(pal_burst_phase(row));
            let b = C::polar(pal_burst_phase(row - 2));
            let c = C::polar(pal_burst_phase(row + 2));
            assert!((a.re - (b.re + c.re) * 0.5).hypot(a.im - (b.im + c.im) * 0.5) > 1.99);
        }
    }
    #[test]
    fn lost_vertical_pulses_preserve_elapsed_pal_fields() {
        let fs = 315e6 / 11.0;
        let period = (fs * 0.02_f64).round() as u64;
        let mut pulses = PulseFinder {
            vertical: vec![10000, 10000 + 3 * period],
            ..Default::default()
        };
        pulses.maintain_field_clock(10000 + 5 * period + (fs * 0.0012) as u64, fs);
        assert_eq!(pulses.vertical.len(), 6);
        assert_eq!(pulses.recovered.len(), 4);
        assert!(pulses
            .vertical
            .windows(2)
            .all(|p| ((p[1] - p[0]) as f64 - fs * 0.02).abs() < 1.0));
    }
    #[test]
    fn cpu_fft_roundtrip_and_spectral_peak() {
        let fft = Fft::new(false);
        let original: Vec<_> = (0..N)
            .map(|i| C {
                re: (std::f32::consts::TAU * 123.0 * i as f32 / N as f32).cos(),
                im: 0.0,
            })
            .collect();
        let mut x = original.clone();
        fft.run(&mut x, false).unwrap();
        assert!((x[123].re - N as f32 / 2.0).abs() < 0.1);
        fft.run(&mut x, true).unwrap();
        assert!(x
            .iter()
            .zip(original)
            .all(|(a, b)| (a.re - b.re).abs() < 1e-5));
    }
    #[test]
    fn pal_playback_filters_keep_dc_shelf_level_and_input_rate_independent_sync_bandwidth() {
        for rate in [20e6, 315e6 / 11.0, 40e6] {
            let dc = pal_video_gain(0.0, rate);
            assert!((dc.re - 1.0).abs() < 1e-5 && dc.im.abs() < 1e-5);
            let mid = pal_video_gain(273_755.82, rate);
            assert!((mid.re.hypot(mid.im) - 1.0 / 5f32.sqrt()).abs() < 1e-4);
            let negative = pal_video_gain(-273_755.82, rate);
            assert!((negative.re - mid.re).abs() < 1e-6);
            assert!((negative.im + mid.im).abs() < 1e-6);
        }
        assert!((pal_sync_gain(0.0) - 1.0).abs() < 1e-6);
        assert!((0.3..0.8).contains(&pal_sync_gain(0.5e6)));
        assert!(pal_sync_gain(1.5e6).abs() < 0.02);
    }

    #[test]
    fn palette_levels_match_pal_tbc_convention() {
        for (hz, expected) in [(3.8e6, 256.0), (4.1e6, 16384.0), (4.8e6, 54016.0)] {
            let value: f64 = 16384.0 + (hz - 4.1e6) * (376.32 / 7000.0);
            assert!((value - expected).abs() < 0.01);
        }
    }
}
