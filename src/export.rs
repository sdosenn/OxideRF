//! Native PAL colour export with OpenCL processing and optional hardware encoding.
//!
//! The 2D filter and burst reference are adapted from PALcolour, GPL-3.0-or-later.
//! Copyright 2018 William Andrew Steer; 2018–2019 Simon Inns; 2019 Adam Sampson.
//! See THIRD_PARTY_NOTICES.md. No upstream executable or metadata converter is used.
use ocl::{Buffer, Kernel, ProQue};
use serde_json::Value;
use std::{
    fs::{self, File},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc, Arc,
    },
    thread::JoinHandle,
    time::Instant,
};

const W: usize = 1135;
const H: usize = 313;
const N: usize = W * H;
const OW: usize = 720;
const OH: usize = 576;
const FRAME_BYTES: usize = OW * OH * 3 / 2;
pub(crate) fn cancelled() -> bool {
    crate::native::stop_requested()
}

struct Capture {
    luma: Option<BufReader<File>>,
    chroma: Option<BufReader<File>>,
    parity: Vec<i32>,
    rf_fields: Vec<f64>,
    start: usize,
    end: usize,
    burst_start: usize,
    burst_end: usize,
    black: f32,
    range: f32,
}
impl Capture {
    fn open(base: &str) -> Result<Self, String> {
        let base = base.strip_suffix(".tbc").unwrap_or(base);
        let metadata =
            File::open(format!("{base}.tbc.json")).map_err(|e| format!("TBC metadata: {e}"))?;
        if metadata.metadata().map_err(|e| e.to_string())?.len() > 256 * 1024 * 1024 {
            return Err("TBC metadata exceeds the 256 MiB safety limit".into());
        }
        let m: Value =
            serde_json::from_reader(BufReader::new(metadata)).map_err(|e| e.to_string())?;
        let v = &m["videoParameters"];
        if v["system"] != "PAL"
            || v["fieldWidth"] != W
            || v["fieldHeight"] != H
            || (v["sampleRate"].as_f64().unwrap_or(0.0) - 17_734_475.0).abs() > 1.0
        {
            return Err(
                "MKV export requires PAL 1135 × 313 TBC at four-fSC (17.734475 MHz)".into(),
            );
        }
        let integer = |key: &str| {
            v[key]
                .as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| format!("Missing TBC parameter: {key}"))
        };
        let start = integer("activeVideoStart")?;
        let end = integer("activeVideoEnd")?;
        let burst_start = integer("colourBurstStart")?;
        let burst_end = integer("colourBurstEnd")?;
        if start < 7 || end <= start || end + 7 > W || burst_start >= burst_end || burst_end > start
        {
            return Err("Invalid active-video or burst bounds in TBC metadata".into());
        }
        let black = v["black16bIre"].as_f64().ok_or("Missing black level")? as f32;
        let white = v["white16bIre"].as_f64().ok_or("Missing white level")? as f32;
        if !(0.0..=65535.0).contains(&black) || !(black + 1.0..=65535.0).contains(&white) {
            return Err("Invalid black/white levels in TBC metadata".into());
        }
        let fields = m["fields"].as_array().ok_or("Missing field metadata")?;
        let parity: Vec<i32> = fields
            .iter()
            .map(|f| {
                f["isFirstField"]
                    .as_bool()
                    .map(|first| i32::from(!first))
                    .ok_or_else(|| "Missing field polarity".to_string())
            })
            .collect::<Result<_, _>>()?;
        if parity.is_empty() {
            return Err("The TBC capture contains no fields".into());
        }
        let luma = File::open(format!("{base}.tbc")).map_err(|e| format!("Luma TBC: {e}"))?;
        let chroma =
            File::open(format!("{base}_chroma.tbc")).map_err(|e| format!("Chroma TBC: {e}"))?;
        let expected = parity.len() as u64 * (N * 2) as u64;
        for (name, file) in [("Luma", &luma), ("Chroma", &chroma)] {
            if file.metadata().map_err(|e| e.to_string())?.len() != expected {
                return Err(format!(
                    "{name} TBC size does not match the {} metadata fields; finish decoding first",
                    parity.len()
                ));
            }
        }
        Ok(Self {
            luma: Some(BufReader::new(luma)),
            chroma: Some(BufReader::new(chroma)),
            rf_fields: fields
                .iter()
                .map(|f| f["fileLoc"].as_f64().unwrap_or(f64::NAN))
                .collect(),
            parity,
            start,
            end,
            burst_start,
            burst_end,
            black,
            range: white - black,
        })
    }
    fn read(&mut self, bytes: &mut [u8], samples: &mut [u16]) -> Result<(), String> {
        self.luma
            .as_mut()
            .ok_or("No file-backed luma input")?
            .read_exact(&mut bytes[..N * 2])
            .map_err(|e| format!("Reading luma: {e}"))?;
        self.chroma
            .as_mut()
            .ok_or("No file-backed chroma input")?
            .read_exact(&mut bytes[N * 2..])
            .map_err(|e| format!("Reading chroma: {e}"))?;
        for (sample, pair) in samples.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *sample = u16::from_le_bytes([pair[0], pair[1]]);
        }
        Ok(())
    }
}

fn coefficients() -> Vec<f32> {
    let radius = 0.5 * 17_734_475.0 / (1_100_000.0 / 0.93);
    let mut c = Vec::with_capacity(32);
    let mut divisor = 0.0;
    for b in 0..=7 {
        for row in [0.0f64, 4.0, 2.0, 6.0] {
            let r = ((b * b) as f64 + row * row).sqrt().min(radius);
            c.push(
                ((1.0 + (std::f64::consts::PI * r / radius).cos()) / if b == 0 { 2.0 } else { 1.0 })
                    as f32,
            );
        }
        divisor += 2.0 * (c[b * 4] + 2.0 * (c[b * 4 + 1] + c[b * 4 + 2] + c[b * 4 + 3]));
    }
    for x in &mut c {
        *x /= divisor;
    }
    c
}
fn sine(x: usize) -> f32 {
    [0.0, 1.0, 0.0, -1.0][x & 3]
}
fn cosine(x: usize) -> f32 {
    [1.0, 0.0, -1.0, 0.0][x & 3]
}
fn chroma(input: &[u16], row: i32, x: usize, active: bool) -> f32 {
    if !(if active { 22..310 } else { 0..H as i32 }).contains(&row) {
        return 0.0;
    }
    input[N + row as usize * W + x] as f32 - 32767.0
}
fn burst(input: &[u16], row: i32, begin: usize, end: usize) -> [f32; 3] {
    let (mut p, mut q, mut po, mut qo) = (0.0, 0.0, 0.0, 0.0);
    for x in begin..end {
        let a = (chroma(input, row, x, false)
            - (chroma(input, row - 2, x, false) + chroma(input, row + 2, x, false)) * 0.5)
            * 0.5;
        let b = (chroma(input, row + 1, x, false) - chroma(input, row - 1, x, false)) * 0.5;
        p += a * sine(x);
        q += a * cosine(x);
        po += b * sine(x);
        qo += b * cosine(x);
    }
    let len = (end - begin) as f32;
    p /= len;
    q /= len;
    po /= len;
    qo /= len;
    let sign = if (p - po).powi(2) + (q - qo).powi(2) < 2.0 * (p * p + q * q) {
        1.0
    } else {
        -1.0
    };
    let (bp, bq) = ((p - qo) * 0.5, (q + po) * 0.5);
    let norm = bp.hypot(bq).max(130000.0 / 128.0);
    [bp / norm, bq / norm, sign]
}
fn component(c: &[f32], plane: usize, x: f32, y: f32, start: usize, end: usize) -> f32 {
    let x = x.clamp(start as f32, (end - 1) as f32);
    let y = y.clamp(22.0, 309.0);
    let (a, b) = (x as usize, y as usize);
    let (r, d) = ((a + 1).min(end - 1), (b + 1).min(309));
    let (tx, ty) = (x - a as f32, y - b as f32);
    let get = |row, col| c[plane * N + row * W + col];
    let top = get(b, a) * (1.0 - tx) + get(b, r) * tx;
    let bottom = get(d, a) * (1.0 - tx) + get(d, r) * tx;
    top * (1.0 - ty) + bottom * ty
}
fn quantize(x: f32) -> u8 {
    x.clamp(1.0, 254.75).round_ties_even() as u8
}

pub(crate) struct Cpu {
    c: Vec<f32>,
    coefficients: Vec<f32>,
}
impl Cpu {
    pub(crate) fn new() -> Self {
        Self {
            c: vec![0.0; N * 3],
            coefficients: coefficients(),
        }
    }
    fn decode(
        &mut self,
        input: &[u16],
        start: usize,
        end: usize,
        burst_start: usize,
        burst_end: usize,
    ) {
        self.decode_rows(input, 22..310, start, end, true, burst_start, burst_end);
    }
    /// PAL colour on every line and sample, so burst and blanking decode too.
    pub(crate) fn decode_full(&mut self, input: &[u16]) {
        self.decode_rows(input, 0..H, 0, W, false, 98, 138);
    }
    #[allow(clippy::too_many_arguments)]
    fn decode_rows(
        &mut self,
        input: &[u16],
        rows: std::ops::Range<usize>,
        start: usize,
        end: usize,
        active: bool,
        burst_start: usize,
        burst_end: usize,
    ) {
        for row in rows {
            let [bp, bq, sign] = burst(input, row as i32, burst_start, burst_end);
            for x in start..end {
                let (mut pu, mut qu, mut pv, mut qv) = (0.0, 0.0, 0.0, 0.0);
                for b in 0..=7 {
                    let (mut m, mut n) = ([0.0; 4], [0.0; 4]);
                    // Taps beyond the left/right raster edge contribute nothing.
                    for i in [x.checked_sub(b), Some(x + b).filter(|&i| i < W)]
                        .into_iter()
                        .flatten()
                    {
                        let r = row as i32;
                        let taps = [
                            chroma(input, r, i, active),
                            -chroma(input, r - 2, i, active) - chroma(input, r + 2, i, active),
                            chroma(input, r - 1, i, active) - chroma(input, r + 1, i, active),
                            -chroma(input, r - 3, i, active) + chroma(input, r + 3, i, active),
                        ];
                        for k in 0..4 {
                            m[k] += taps[k] * sine(i);
                            n[k] += taps[k] * cosine(i);
                        }
                    }
                    let f = &self.coefficients[b * 4..b * 4 + 4];
                    pu += m[0] * f[0] + m[1] * f[1] + n[2] * f[2] + n[3] * f[3];
                    qu += n[0] * f[0] + n[1] * f[1] - m[2] * f[2] - m[3] * f[3];
                    pv += m[0] * f[0] + m[1] * f[1] - n[2] * f[2] - n[3] * f[3];
                    qv += n[0] * f[0] + n[1] * f[1] + m[2] * f[2] + m[3] * f[3];
                }
                let at = row * W + x;
                self.c[at] = input[at] as f32;
                self.c[N + at] = -2.0 * (pu * bp + qu * bq);
                self.c[2 * N + at] = -2.0 * sign * (qv * bp - pv * bq);
            }
        }
    }
    pub(crate) fn preview(&mut self, input: &[u16]) -> Vec<u8> {
        self.decode(input, 185, 1107, 98, 138);
        let width = crate::preview::WIDTH;
        let height = crate::preview::HEIGHT;
        let range = crate::preview::WHITE - crate::preview::BLACK;
        let get = |plane, x: usize, y: usize| {
            let sx = 185.0 + (x as f32 + 0.5) * 922.0 / width as f32 - 0.5;
            component(&self.c, plane, sx, 22.0 + y as f32, 185, 1107)
        };
        let mut out = vec![0; width * height * 3 / 2];
        for y in 0..height {
            for x in 0..width {
                out[y * width + x] =
                    quantize(16.0 + (get(0, x, y) - crate::preview::BLACK) * 219.0 / range);
                if x % 2 == 0 && y % 2 == 0 {
                    let (mut u, mut v) = (0.0, 0.0);
                    for dy in 0..2 {
                        for dx in 0..2 {
                            u += get(1, x + dx, y + dy);
                            v += get(2, x + dx, y + dy);
                        }
                    }
                    let at = width * height + y / 2 * width + x;
                    out[at] = quantize(128.0 + u * 0.25 * 112.0 / (0.886 * 0.492_111_03 * range));
                    out[at + 1] =
                        quantize(128.0 + v * 0.25 * 112.0 / (0.701 * 0.877_283_2 * range));
                }
            }
        }
        out
    }
    /// NV12 colour preview of the full TBC raster, `width` columns × FULL_HEIGHT
    /// lines; same output as the OpenCL `preview_full_nv12` kernel.
    pub(crate) fn preview_full(
        &mut self,
        input: &[u16],
        width: usize,
        mode: crate::preview::RasterChroma,
    ) -> Vec<u8> {
        use crate::preview::RasterChroma;
        match mode {
            RasterChroma::Active => self.decode(input, 185, 1107, 98, 138),
            RasterChroma::Whole => self.decode_full(input),
            RasterChroma::Composite => {}
        }
        let height = crate::preview::FULL_HEIGHT;
        let range = crate::preview::WHITE - crate::preview::BLACK;
        let column = |x: usize| (x as f32 + 0.5) * W as f32 / width as f32 - 0.5;
        let luma = |x: usize, y: usize| {
            if y >= H {
                return crate::preview::BLACK;
            }
            let sx = column(x).clamp(0.0, (W - 1) as f32);
            let a = sx as usize;
            let b = (a + 1).min(W - 1);
            let t = sx - a as f32;
            let sample = |plane: usize| {
                let at = plane * N + y * W;
                input[at + a] as f32 * (1.0 - t) + input[at + b] as f32 * t
            };
            if mode == RasterChroma::Composite {
                sample(0) + sample(1) - 32767.0
            } else {
                sample(0)
            }
        };
        let chroma = |plane: usize, x: usize, y: usize| {
            let sx = column(x);
            match mode {
                RasterChroma::Composite => 0.0,
                _ if y >= H => 0.0,
                RasterChroma::Whole => {
                    let sx = sx.clamp(0.0, (W - 1) as f32);
                    let a = sx as usize;
                    let b = (a + 1).min(W - 1);
                    let t = sx - a as f32;
                    self.c[plane * N + y * W + a] * (1.0 - t) + self.c[plane * N + y * W + b] * t
                }
                RasterChroma::Active => {
                    if !(22..310).contains(&y) || !(185.0..=1106.0).contains(&sx) {
                        return 0.0;
                    }
                    component(&self.c, plane, sx, y as f32, 185, 1107)
                }
            }
        };
        let mut out = vec![0; width * height * 3 / 2];
        for y in 0..height {
            for x in 0..width {
                out[y * width + x] =
                    quantize(16.0 + (luma(x, y) - crate::preview::BLACK) * 219.0 / range);
                if x % 2 == 0 && y % 2 == 0 {
                    let (mut u, mut v) = (0.0, 0.0);
                    for dy in 0..2 {
                        for dx in 0..2 {
                            u += chroma(1, x + dx, y + dy);
                            v += chroma(2, x + dx, y + dy);
                        }
                    }
                    let at = width * height + y / 2 * width + x;
                    out[at] = quantize(128.0 + u * 0.25 * 112.0 / (0.886 * 0.492_111_03 * range));
                    out[at + 1] =
                        quantize(128.0 + v * 0.25 * 112.0 / (0.701 * 0.877_283_2 * range));
                }
            }
        }
        out
    }
    fn field(&mut self, input: &[u16], capture: &Capture, parity: i32, out: &mut [u8]) {
        self.decode(
            input,
            capture.start,
            capture.end,
            capture.burst_start,
            capture.burst_end,
        );
        let get = |p, x: usize, y: usize| {
            let sx = capture.start as f32
                + (x as f32 + 0.5) * (capture.end - capture.start) as f32 / OW as f32
                - 0.5;
            let sy = (44.0 + y as f32 - parity as f32) * 0.5;
            component(&self.c, p, sx, sy, capture.start, capture.end)
        };
        for y in 0..OH {
            for x in 0..OW {
                out[y * OW + x] =
                    quantize(16.0 + (get(0, x, y) - capture.black) * 219.0 / capture.range);
                if x % 2 == 0 && y % 2 == 0 {
                    let (mut u, mut v) = (0.0, 0.0);
                    for dy in 0..2 {
                        for dx in 0..2 {
                            u += get(1, x + dx, y + dy);
                            v += get(2, x + dx, y + dy);
                        }
                    }
                    let at = OW * OH + y / 2 * OW + x;
                    out[at] =
                        quantize(128.0 + u * 0.25 * 112.0 / (0.886 * 0.492_111_03 * capture.range));
                    out[at + 1] =
                        quantize(128.0 + v * 0.25 * 112.0 / (0.701 * 0.877_283_2 * capture.range));
                }
            }
        }
    }
}

pub(crate) struct GpuPreview {
    burst: Kernel,
    pal: Kernel,
    render: Kernel,
    output: Buffer<u8>,
    bytes: usize,
    _work: Vec<Buffer<f32>>,
}
impl GpuPreview {
    /// Colour preview of the active 4:3 picture, or with `full_width` of the
    /// full TBC raster scaled to that many columns.
    pub(crate) fn new(
        pq: &ProQue,
        input: &Buffer<u16>,
        full: Option<(usize, crate::preview::RasterChroma)>,
    ) -> ocl::Result<Self> {
        let bytes = match full {
            Some((width, _)) => width * crate::preview::FULL_HEIGHT * 3 / 2,
            None => OW * 288 * 3 / 2,
        };
        let whole = matches!(full, Some((_, crate::preview::RasterChroma::Whole)));
        let allocate = |len| {
            Buffer::<f32>::builder()
                .queue(pq.queue().clone())
                .len(len)
                .build()
        };
        let burst_data = allocate(H * 3)?;
        let component = allocate(N * 3)?;
        let coeff = Buffer::builder()
            .queue(pq.queue().clone())
            .len(32)
            .copy_host_slice(&coefficients())
            .build()?;
        let output = Buffer::<u8>::builder()
            .queue(pq.queue().clone())
            .len(bytes)
            .build()?;
        let burst = pq
            .kernel_builder("export_burst")
            .global_work_size(H)
            .arg(input)
            .arg(&burst_data)
            .arg(98i32)
            .arg(138i32)
            .build()?;
        let pal = if whole {
            pq.kernel_builder("export_pal_full")
                .global_work_size([W, H])
                .arg(input)
                .arg(&burst_data)
                .arg(&coeff)
                .arg(&component)
                .build()?
        } else {
            pq.kernel_builder("export_pal")
                .global_work_size([922, 288])
                .arg(input)
                .arg(&burst_data)
                .arg(&coeff)
                .arg(&component)
                .arg(185i32)
                .arg(1107i32)
                .build()?
        };
        let render = match full {
            Some((width, mode)) => pq
                .kernel_builder("preview_full_nv12")
                .global_work_size([width, crate::preview::FULL_HEIGHT])
                .arg(input)
                .arg(&component)
                .arg(&output)
                .arg(width as i32)
                .arg(mode.code())
                .build()?,
            None => pq
                .kernel_builder("preview_colour_nv12")
                .global_work_size([OW, 288])
                .arg(&component)
                .arg(&output)
                .build()?,
        };
        Ok(Self {
            burst,
            pal,
            render,
            output,
            bytes,
            _work: vec![burst_data, component, coeff],
        })
    }
    pub(crate) fn render(&self) -> ocl::Result<Vec<u8>> {
        // PAL field dimensions, crop and burst bounds are fixed and validated.
        unsafe {
            self.burst.enq()?;
            self.pal.enq()?;
            self.render.enq()?;
        }
        let mut out = vec![0u8; self.bytes];
        self.output.read(&mut out).enq()?;
        Ok(out)
    }
}

struct Gpu {
    _pq: ProQue,
    input: Buffer<u16>,
    output: Buffer<u8>,
    burst: Kernel,
    pal: Kernel,
    nv12: Kernel,
    _burst: Buffer<f32>,
    _component: Buffer<f32>,
    _coefficients: Buffer<f32>,
}
impl Gpu {
    fn components(&self, input: &[u16], output: &mut [f32]) -> Result<(), String> {
        let run = || -> ocl::Result<()> {
            self.input.write(input).enq()?;
            // The in-order queue completes PAL separation before host packing.
            unsafe {
                self.burst.enq()?;
                self.pal.enq()?;
            }
            self._component.read(output).enq()?;
            Ok(())
        };
        run().map_err(|e| e.to_string())
    }
    fn new(c: &Capture, device: usize) -> Result<Self, String> {
        Self::with_region(c, device, false)
    }
    /// With `whole`, PAL colour is decoded on every line and sample.
    fn with_region(c: &Capture, device: usize, whole: bool) -> Result<Self, String> {
        let mut devices = Vec::new();
        for platform in crate::platforms() {
            for d in
                ocl::Device::list(platform, Some(ocl::flags::DEVICE_TYPE_GPU)).unwrap_or_default()
            {
                devices.push((platform, d));
            }
        }
        let (platform, d) = devices
            .get(device)
            .ok_or("No selected OpenCL GPU available for colour export")?;
        let build = || -> ocl::Result<Self> {
            let pq = ProQue::builder()
                .platform(*platform)
                .device(*d)
                .src(include_str!("../kernels/export.cl"))
                .dims(1)
                .build()?;
            let input = Buffer::builder()
                .queue(pq.queue().clone())
                .len(N * 2)
                .build()?;
            let output = Buffer::builder()
                .queue(pq.queue().clone())
                .len(FRAME_BYTES)
                .build()?;
            let row_burst = Buffer::<f32>::builder()
                .queue(pq.queue().clone())
                .len(H * 3)
                .build()?;
            let component = Buffer::<f32>::builder()
                .queue(pq.queue().clone())
                .len(N * 3)
                .build()?;
            let coeff = Buffer::builder()
                .queue(pq.queue().clone())
                .len(32)
                .copy_host_slice(&coefficients())
                .build()?;
            let burst = pq
                .kernel_builder("export_burst")
                .global_work_size(H)
                .arg(&input)
                .arg(&row_burst)
                .arg(c.burst_start as i32)
                .arg(c.burst_end as i32)
                .build()?;
            let pal = if whole {
                pq.kernel_builder("export_pal_full")
                    .global_work_size([W, H])
                    .arg(&input)
                    .arg(&row_burst)
                    .arg(&coeff)
                    .arg(&component)
                    .build()?
            } else {
                pq.kernel_builder("export_pal")
                    .global_work_size([c.end - c.start, 288])
                    .arg(&input)
                    .arg(&row_burst)
                    .arg(&coeff)
                    .arg(&component)
                    .arg(c.start as i32)
                    .arg(c.end as i32)
                    .build()?
            };
            let nv12 = pq
                .kernel_builder("export_nv12")
                .global_work_size([OW, OH])
                .arg(&component)
                .arg(&output)
                .arg(c.start as i32)
                .arg(c.end as i32)
                .arg(0i32)
                .arg(c.black)
                .arg(c.range)
                .build()?;
            println!("EXPORT_OPENCL {}", d.name().unwrap_or_default());
            Ok(Self {
                _pq: pq,
                input,
                output,
                burst,
                pal,
                nv12,
                _burst: row_burst,
                _component: component,
                _coefficients: coeff,
            })
        };
        build().map_err(|e| e.to_string())
    }
    fn field(&self, input: &[u16], parity: i32, out: &mut [u8]) -> Result<(), String> {
        let run = || -> ocl::Result<()> {
            self.input.write(input).enq()?;
            self.nv12.set_arg(4, parity)?;
            // The kernels use validated fixed-size PAL fields and an in-order queue.
            unsafe {
                self.burst.enq()?;
                self.pal.enq()?;
                self.nv12.enq()?;
            }
            self.output.read(out).enq()?;
            Ok(())
        };
        run().map_err(|e| e.to_string())
    }
}

#[derive(Clone, Debug)]
enum Encoder {
    Cpu,
    Vaapi(PathBuf),
    Nvenc,
}
impl Encoder {
    fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU H.264",
            Self::Vaapi(_) => "VAAPI GPU H.264",
            Self::Nvenc => "NVENC GPU H.264",
        }
    }
    fn input_args(&self, cmd: &mut Command) {
        if let Self::Vaapi(device) = self {
            cmd.arg("-vaapi_device").arg(device);
        }
    }
    fn raw_input(&self, cmd: &mut Command) {
        self.input_args(cmd);
        cmd.args([
            "-f",
            "rawvideo",
            "-pixel_format",
            "nv12",
            "-video_size",
            "720x576",
            "-framerate",
            "50",
        ]);
        // Set decoder-side colour properties before hwupload. Otherwise newer
        // FFmpeg versions can insert an unsupported software conversion after it.
        Self::colour_flags(cmd);
        cmd.args(["-i", "pipe:0"]);
    }
    fn colour_flags(cmd: &mut Command) {
        cmd.args([
            "-color_range",
            "tv",
            "-colorspace",
            "bt470bg",
            "-color_primaries",
            "bt470bg",
            "-color_trc",
            "gamma28",
        ]);
    }
    fn output_args(&self, cmd: &mut Command) {
        match self {
            Self::Cpu => {
                cmd.args(["-c:v", "libx264", "-preset", "fast", "-crf", "18"]);
            }
            Self::Vaapi(_) => {
                cmd.args([
                    "-vf",
                    "format=nv12,hwupload",
                    "-c:v",
                    "h264_vaapi",
                    "-qp",
                    "20",
                ]);
            }
            Self::Nvenc => {
                cmd.args([
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p4",
                    "-rc",
                    "vbr",
                    "-cq",
                    "19",
                    "-b:v",
                    "0",
                ]);
            }
        }
    }
    fn probe(&self, ffmpeg: &Path) -> Result<(), String> {
        let mut cmd = Command::new(ffmpeg);
        cmd.args(["-hide_banner", "-loglevel", "error", "-nostdin"]);
        self.raw_input(&mut cmd);
        cmd.args(["-frames:v", "2"]);
        self.output_args(&mut cmd);
        Self::colour_flags(&mut cmd);
        let mut child = cmd
            .args(["-f", "null", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                format!("Cannot launch FFmpeg: {e}. Install FFmpeg with H.264 support.")
            })?;
        if let Some(mut pipe) = child.stdin.take() {
            let mut black = vec![128u8; FRAME_BYTES];
            black[..OW * OH].fill(16);
            // A rejected encoder may close the pipe early; its status/stderr
            // below provide the useful diagnostic instead of a pipe error.
            let _ = pipe.write_all(&black).and_then(|()| pipe.write_all(&black));
        }
        let result = child.wait_with_output().map_err(|e| e.to_string())?;
        if result.status.success() {
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&result.stderr).trim().to_string())
        }
    }
}
fn choose_encoder(preference: &str, ffmpeg: &Path) -> Result<Encoder, String> {
    if !["auto", "gpu", "cpu"].contains(&preference) {
        return Err("Encoder must be auto, gpu or cpu".into());
    }
    // Test actual encoding, not just whether FFmpeg lists a codec wrapper.
    if preference != "cpu" {
        let mut nodes: Vec<PathBuf> = fs::read_dir("/dev/dri")
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().starts_with("renderD"))
            .map(|e| e.path())
            .collect();
        nodes.sort();
        let mut candidates: Vec<Encoder> = nodes.into_iter().map(Encoder::Vaapi).collect();
        candidates.push(Encoder::Nvenc);
        for encoder in candidates {
            match encoder.probe(ffmpeg) {
                Ok(()) => return Ok(encoder),
                Err(e) => eprintln!("EXPORT_ENCODER_UNAVAILABLE {}: {e}", encoder.name()),
            }
        }
        if preference == "gpu" {
            return Err("No working VAAPI/NVENC H.264 encoder. Check FFmpeg, render-device permissions and graphics drivers, or select Automatic/CPU.".into());
        }
        println!("EXPORT_ENCODER_FALLBACK GPU encoding unavailable; using CPU H.264");
    }
    Encoder::Cpu.probe(ffmpeg)?;
    Ok(Encoder::Cpu)
}
struct EncodingChild(Child);
impl Drop for EncodingChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Export matching PAL luma/chroma TBC files using the native colour decoder.
pub fn run(args: &[String]) -> Result<(), String> {
    let wall = Instant::now();
    let get = |key: &str, default: &str| {
        args.windows(2)
            .find(|w| w[0] == key)
            .map(|w| w[1].clone())
            .unwrap_or_else(|| default.into())
    };
    let input = get("--input", "");
    let output = get(
        "--output",
        &format!("{}.mkv", input.strip_suffix(".tbc").unwrap_or(&input)),
    );
    if input.is_empty() {
        return Err("Use --export-mkv --input TBC_BASE --output VIDEO.mkv [--backend opencl|cpu] [--encoder auto|gpu|cpu]".into());
    }
    match get("--profile", "viewing").as_str() {
        "archive" => return run_archive(args),
        "viewing" => {}
        _ => return Err("Export profile must be archive or viewing".into()),
    }
    if Path::new(&output).exists() {
        return Err(format!(
            "Output already exists: {output}. Choose another name."
        ));
    }
    let mut capture = Capture::open(&input)?;
    let backend = get("--backend", "opencl");
    if !["opencl", "cpu"].contains(&backend.as_str()) {
        return Err("Export backend must be opencl or cpu".into());
    }
    let device = get("--device", "0")
        .parse()
        .map_err(|_| "Invalid OpenCL device index")?;
    let frames = get("--frames", "0")
        .parse::<usize>()
        .map_err(|_| "Invalid frame count")?;
    let count = if frames == 0 {
        capture.parity.len()
    } else {
        frames.saturating_mul(2).min(capture.parity.len())
    };
    crate::native::install_stop_handler()?;
    let ffmpeg = PathBuf::from(std::env::var_os("VHS_FFMPEG").unwrap_or_else(|| "ffmpeg".into()));
    let encoder = choose_encoder(&get("--encoder", "auto"), &ffmpeg)?;
    let gpu = if backend == "opencl" {
        Some(Gpu::new(&capture, device)?)
    } else {
        None
    };
    let mut cpu = if gpu.is_none() {
        Some(Cpu::new())
    } else {
        None
    };
    if cancelled() {
        return Ok(());
    }
    let mut cmd = Command::new(ffmpeg);
    cmd.args(["-hide_banner", "-loglevel", "warning", "-nostdin", "-n"]);
    encoder.raw_input(&mut cmd);
    let audio = crate::audio::export_track(
        &input,
        Path::new(&output),
        &capture.rf_fields,
        &(0..count).collect::<Vec<_>>(),
        get("--audio-offset-ms", "0")
            .parse()
            .map_err(|_| "Invalid audio offset")?,
        true,
    )?;
    crate::audio::ExportTrack::configure(audio.as_ref(), &mut cmd);
    encoder.output_args(&mut cmd);
    Encoder::colour_flags(&mut cmd);
    cmd.args(["-aspect", "4:3", "-f", "matroska"])
        .arg(&output)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    let mut child = EncodingChild(cmd.spawn().map_err(|e| format!("FFmpeg: {e}"))?);
    let mut pipe = child.0.stdin.take().ok_or("FFmpeg stdin unavailable")?;
    println!(
        "EXPORT_START encoder={} · colour={} · 720×576 PAL 50p · audio={}",
        encoder.name(),
        backend,
        audio.is_some()
    );
    let began = Instant::now();
    let (mut bytes, mut samples, mut nv12) =
        (vec![0u8; N * 4], vec![0u16; N * 2], vec![0u8; FRAME_BYTES]);
    let mut done = 0;
    for index in 0..count {
        if cancelled() {
            break;
        }
        capture.read(&mut bytes, &mut samples)?;
        let parity = capture.parity[index];
        if let Some(gpu) = &gpu {
            gpu.field(&samples, parity, &mut nv12)?;
        } else {
            cpu.as_mut()
                .unwrap()
                .field(&samples, &capture, parity, &mut nv12);
        }
        pipe.write_all(&nv12).map_err(|e|format!("FFmpeg input failed: {e}. See encoder messages; any existing partial MKV is retained."))?;
        done += 1;
        if done % 50 == 0 || done == count {
            println!(
                "EXPORT_PROGRESS fields={done}/{count} percent={:.1} speed={:.1} fields/s",
                done as f64 * 100.0 / count as f64,
                done as f64 / began.elapsed().as_secs_f64().max(0.001)
            );
        }
    }
    drop(pipe);
    let status = child.0.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!(
            "FFmpeg encoding failed ({status}); any partial MKV is retained: {output}"
        ));
    }
    println!(
        "EXPORT_OK fields={done} duration_s={:.3} stopped={} wall_s={:.3} realtime={:.2}x output={output}",
        done as f64 / 50.0,
        cancelled(),wall.elapsed().as_secs_f64(),done as f64/50.0/wall.elapsed().as_secs_f64().max(0.001)
    );
    Ok(())
}

/// Pack original PAL lines into a 10-bit planar 4:2:2 interlaced frame.
/// Horizontal padding preserves all active samples without rescaling to 720.
struct ArchiveFrame {
    width: usize,
    height: usize,
    /// Every TBC sample and line instead of the active picture.
    full: bool,
    mode: crate::preview::RasterChroma,
    data: Vec<u8>,
    first: bool,
}
/// Full-raster MKV frame: 1135 TBC samples padded to an even 4:2:2 width, and
/// both 313-line fields.
const FULL_W: usize = W + 1;
const FULL_H: usize = H * 2;
impl ArchiveFrame {
    fn new(capture: &Capture) -> Self {
        Self::sized((capture.end - capture.start).div_ceil(8) * 8, OH, false)
    }
    fn full(mode: crate::preview::RasterChroma) -> Self {
        Self {
            mode,
            ..Self::sized(FULL_W, FULL_H, true)
        }
    }
    fn sized(width: usize, height: usize, full: bool) -> Self {
        let mut data = vec![0; width * height * 4];
        for (index, word) in data.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            word.copy_from_slice(
                &(if index < width * height {
                    64u16
                } else {
                    512u16
                })
                .to_le_bytes(),
            );
        }
        Self {
            width,
            height,
            full,
            mode: crate::preview::RasterChroma::Active,
            data,
            first: false,
        }
    }
    /// Full raster: luma straight from the TBC samples on every line, including
    /// sync, blanking and VBI (levels below code 4 clip). Chroma follows the
    /// mode: active picture only, the whole raster, or none with the raw
    /// chroma signal added to luma (composite).
    fn field_full(
        &mut self,
        components: &[f32],
        samples: &[u16],
        c: &Capture,
        parity: i32,
    ) -> bool {
        if parity == 0 {
            self.first = true;
        } else if !self.first {
            return false;
        }
        let put = |data: &mut [u8], index: usize, value: f32| {
            let value = value.clamp(4.0, 1019.0).round_ties_even() as u16;
            data[index * 2..index * 2 + 2].copy_from_slice(&value.to_le_bytes());
        };
        use crate::preview::RasterChroma;
        let half = self.width / 2;
        let composite = self.mode == RasterChroma::Composite;
        for row in 0..H {
            let y = row * 2 + parity as usize;
            for x in 0..W {
                let mut luma = samples[row * W + x] as f32;
                if composite {
                    luma += samples[N + row * W + x] as f32 - 32767.0;
                }
                put(
                    &mut self.data,
                    y * self.width + x,
                    64.0 + (luma - c.black) * 876.0 / c.range,
                );
            }
            let active = (22..310).contains(&row);
            for x in 0..half {
                let mut uv = [0.0; 2];
                for dx in 0..2 {
                    let px = x * 2 + dx;
                    let coloured = match self.mode {
                        RasterChroma::Active => active && (c.start..c.end).contains(&px),
                        RasterChroma::Whole => px < W,
                        RasterChroma::Composite => false,
                    };
                    if coloured {
                        uv[0] += components[N + row * W + px] * 0.5;
                        uv[1] += components[2 * N + row * W + px] * 0.5;
                    }
                }
                for (plane, scale) in [0.886 * 0.492_111_03, 0.701 * 0.877_283_2]
                    .iter()
                    .enumerate()
                {
                    let at = self.width * self.height + plane * half * self.height + y * half + x;
                    put(
                        &mut self.data,
                        at,
                        512.0 + uv[plane] * 448.0 / (scale * c.range),
                    );
                }
            }
        }
        if parity == 1 {
            self.first = false;
            true
        } else {
            false
        }
    }
    fn field(&mut self, components: &[f32], c: &Capture, parity: i32) -> bool {
        if parity == 0 {
            self.first = true;
        } else if !self.first {
            return false;
        }
        let active = c.end - c.start;
        let left = (self.width - active) / 2;
        let put = |data: &mut [u8], index: usize, value: f32| {
            let value = value.clamp(4.0, 1019.0).round_ties_even() as u16;
            data[index * 2..index * 2 + 2].copy_from_slice(&value.to_le_bytes());
        };
        for row in 0..288 {
            let y = row * 2 + parity as usize;
            for x in 0..active {
                let at = (row + 22) * W + c.start + x;
                put(
                    &mut self.data,
                    y * self.width + left + x,
                    64.0 + (components[at] - c.black) * 876.0 / c.range,
                );
            }
            for x in 0..self.width / 2 {
                let mut uv = [0.0; 2];
                for dx in 0..2 {
                    let px = x * 2 + dx;
                    if px >= left && px - left < active {
                        let at = (row + 22) * W + c.start + px - left;
                        uv[0] += components[N + at] * 0.5;
                        uv[1] += components[2 * N + at] * 0.5;
                    }
                }
                for (plane, scale) in [0.886 * 0.492_111_03, 0.701 * 0.877_283_2]
                    .iter()
                    .enumerate()
                {
                    let at = self.width * OH + plane * self.width / 2 * OH + y * self.width / 2 + x;
                    put(
                        &mut self.data,
                        at,
                        512.0 + uv[plane] * 448.0 / (scale * c.range),
                    );
                }
            }
        }
        if parity == 1 {
            self.first = false;
            true
        } else {
            false
        }
    }
}

fn run_archive(args: &[String]) -> Result<(), String> {
    let get = |key: &str, default: &str| {
        args.windows(2)
            .find(|w| w[0] == key)
            .map(|w| w[1].clone())
            .unwrap_or_else(|| default.into())
    };
    let input = get("--input", "");
    let output = get(
        "--output",
        &format!("{}.mkv", input.strip_suffix(".tbc").unwrap_or(&input)),
    );
    if Path::new(&output).exists() {
        return Err(format!("Output already exists: {output}"));
    }
    let mut capture = Capture::open(&input)?;
    let backend = get("--backend", "opencl");
    if !["opencl", "cpu"].contains(&backend.as_str()) {
        return Err("Export backend must be opencl or cpu".into());
    }
    let device = get("--device", "0")
        .parse()
        .map_err(|_| "Invalid OpenCL device index")?;
    let gpu = if backend == "opencl" {
        Some(Gpu::new(&capture, device)?)
    } else {
        None
    };
    let mut cpu = Cpu::new();
    let limit = get("--frames", "0")
        .parse::<usize>()
        .map_err(|_| "Invalid frame count")?;
    let mut frame = ArchiveFrame::new(&capture);
    crate::native::install_stop_handler()?;
    let mut indices = Vec::new();
    let mut first = None;
    for (index, &parity) in capture.parity.iter().enumerate() {
        if parity == 0 {
            first = Some(index);
        } else if let Some(first) = first.take() {
            indices.extend([first, index]);
            if limit > 0 && indices.len() / 2 >= limit {
                break;
            }
        }
    }
    let audio = crate::audio::export_track(
        &input,
        Path::new(&output),
        &capture.rf_fields,
        &indices,
        get("--audio-offset-ms", "0")
            .parse()
            .map_err(|_| "Invalid audio offset")?,
        true,
    )?;
    let ffmpeg = std::env::var_os("VHS_FFMPEG").unwrap_or_else(|| "ffmpeg".into());
    let mut cmd = Command::new(ffmpeg);
    cmd.args([
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "yuv422p10le",
        "-video_size",
        &format!("{}x576", frame.width),
        "-framerate",
        "25",
    ]);
    Encoder::colour_flags(&mut cmd);
    cmd.args(["-i", "pipe:0"]);
    crate::audio::ExportTrack::configure(audio.as_ref(), &mut cmd);
    archive_output_args(&mut cmd);
    Encoder::colour_flags(&mut cmd);
    let mut child = EncodingChild(
        cmd.args(["-f", "matroska"])
            .arg(&output)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("FFmpeg: {e}"))?,
    );
    let mut pipe = child.0.stdin.take().ok_or("FFmpeg stdin unavailable")?;
    println!("EXPORT_START encoder=CPU_FFV1 colour={backend} profile=archive format=yuv422p10le size=720x576 rate=25 interlaced=TFF audio={}",audio.is_some());
    let (mut bytes, mut samples, mut components) =
        (vec![0; N * 4], vec![0; N * 2], vec![0.0; N * 3]);
    let began = Instant::now();
    let mut done = 0;
    for index in 0..capture.parity.len() {
        if crate::native::stop_requested() || (limit > 0 && done >= limit) {
            break;
        }
        capture.read(&mut bytes, &mut samples)?;
        if let Some(gpu) = &gpu {
            gpu.components(&samples, &mut components)?;
        } else {
            cpu.decode(
                &samples,
                capture.start,
                capture.end,
                capture.burst_start,
                capture.burst_end,
            );
            components.copy_from_slice(&cpu.c);
        }
        if frame.field(&components, &capture, capture.parity[index]) {
            pipe.write_all(&frame.data)
                .map_err(|e| format!("FFmpeg input failed: {e}. Partial MKV retained."))?;
            done += 1;
            if done % 25 == 0 {
                println!(
                    "EXPORT_PROGRESS frames={done} speed={:.1} frames/s",
                    done as f64 / began.elapsed().as_secs_f64().max(0.001)
                );
            }
        }
    }
    drop(pipe);
    let status = child.0.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!(
            "FFmpeg encoding failed ({status}); partial MKV retained: {output}"
        ));
    }
    if done == 0 {
        return Err("No complete first/second field pairs available for archival export".into());
    }
    println!(
        "EXPORT_OK frames={done} duration_s={:.3} stopped={} wall_s={:.3} output={output}",
        done as f64 / 25.0,
        crate::native::stop_requested(),
        began.elapsed().as_secs_f64()
    );
    Ok(())
}

/// FFV1 for a full-raster frame: no scaling; the sample aspect makes the
/// active 922 × 576 area display at 4:3.
fn full_raster_output_args(cmd: &mut Command) {
    cmd.args([
        "-vf",
        "setfield=tff,setsar=sar=384/461:max=1000",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-slices",
        "16",
        "-slicecrc",
        "1",
        "-threads",
        "4",
        "-pix_fmt",
        "yuv422p10le",
        "-flags",
        "+ilme+ildct",
        "-field_order",
        "tt",
    ]);
}

fn archive_output_args(cmd: &mut Command) {
    cmd.args([
        "-vf",
        "setfield=tff,scale=720:576:flags=lanczos:interl=1,setsar=16/15",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-slices",
        "16",
        "-slicecrc",
        "1",
        "-threads",
        "4",
        "-pix_fmt",
        "yuv422p10le",
        "-flags",
        "+ilme+ildct",
        "-field_order",
        "tt",
        "-aspect",
        "4:3",
    ]);
}

/// Feed decoded fields directly to the existing colour renderer and FFmpeg.
/// Only one field, one assembled frame and the OS pipe are buffered here;
/// encoder backpressure reaches the bounded RF acquisition buffer.
pub(crate) struct DirectMkv {
    base: PathBuf,
    output: PathBuf,
    /// Hidden file written during capture, renamed to `output` once encoding
    /// stops. File managers neither list nor keep re-thumbnailing it while it grows.
    partial: Option<PathBuf>,
    capture: Capture,
    gpu: Option<Gpu>,
    cpu: Cpu,
    archive: Option<ArchiveFrame>,
    components: Vec<f32>,
    viewing: Vec<u8>,
    encoder: Option<EncodingChild>,
    pipe: Option<std::process::ChildStdin>,
    first_rf: Option<f64>,
    retained_rf: Vec<f64>,
    audio_offset: f64,
    frames: usize,
    stats: Arc<DirectStats>,
    /// Decoder field number of the pending archive frame's first field.
    first_field: Option<u64>,
    /// Burns teletext header images from the VBI receiver into the video.
    burn: Option<crate::burn::Burner>,
}
/// Cumulative encoder-thread time, read by live diagnostics.
#[derive(Default)]
pub(crate) struct DirectStats {
    pub render_ns: AtomicU64,
    pub write_ns: AtomicU64,
    pub send_ns: AtomicU64,
}
impl DirectMkv {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        base: &Path,
        profile: &str,
        preference: &str,
        gpu: bool,
        device: usize,
        audio_offset: f64,
        full_raster: Option<crate::preview::RasterChroma>,
        burn_header: Option<crate::preview::HeaderPlacement>,
    ) -> Result<Self, String> {
        if !["archive", "viewing"].contains(&profile) {
            return Err("MKV profile must be archive or viewing".into());
        }
        if !audio_offset.is_finite() || audio_offset.abs() > 10000.0 {
            return Err("Audio offset must be within ±10000 ms".into());
        }
        let output = PathBuf::from(format!("{}.mkv", base.display()));
        if output.exists() {
            return Err(format!("Output already exists: {}", output.display()));
        }
        let partial = partial_path(&output)?;
        if partial.exists() {
            return Err(format!(
                "An unfinished recording already exists: {}. Rename or delete it first.",
                partial.display()
            ));
        }
        if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let capture = Capture {
            luma: None,
            chroma: None,
            parity: Vec::new(),
            rf_fields: Vec::new(),
            start: 185,
            end: 1107,
            burst_start: 98,
            burst_end: 138,
            black: crate::preview::BLACK,
            range: crate::preview::WHITE - crate::preview::BLACK,
        };
        let gpu = if gpu {
            let whole = full_raster == Some(crate::preview::RasterChroma::Whole);
            Some(Gpu::with_region(&capture, device, whole)?)
        } else {
            None
        };
        // Full raster is an archival/analysis format: always lossless FFV1.
        let profile = if full_raster.is_some() {
            "archive"
        } else {
            profile
        };
        let archive = if let Some(mode) = full_raster {
            Some(ArchiveFrame::full(mode))
        } else {
            (profile == "archive").then(|| ArchiveFrame::new(&capture))
        };
        let burn = burn_header.map(|placement| {
            use crate::burn::{header_rect, Burner, Layout};
            let active = capture.end - capture.start;
            match &archive {
                Some(frame) => Burner::new(
                    Layout::Yuv422p10 {
                        width: frame.width,
                        height: frame.height,
                    },
                    if frame.full {
                        header_rect(capture.start, active, 44, 576, placement)
                    } else {
                        header_rect((frame.width - active) / 2, active, 0, 576, placement)
                    },
                ),
                None => Burner::new(
                    Layout::Nv12 {
                        width: OW,
                        height: OH,
                    },
                    header_rect(0, OW, 0, OH, placement),
                ),
            }
        });
        let ffmpeg =
            PathBuf::from(std::env::var_os("VHS_FFMPEG").unwrap_or_else(|| "ffmpeg".into()));
        let mut cmd = Command::new(&ffmpeg);
        cmd.args(["-hide_banner", "-loglevel", "warning", "-nostdin", "-n"]);
        let name;
        if let Some(frame) = &archive {
            cmd.args([
                "-f",
                "rawvideo",
                "-pixel_format",
                "yuv422p10le",
                "-video_size",
                &format!("{}x{}", frame.width, frame.height),
                "-framerate",
                "25",
            ]);
            Encoder::colour_flags(&mut cmd);
            cmd.args(["-i", "pipe:0", "-an"]);
            if frame.full {
                full_raster_output_args(&mut cmd);
            } else {
                archive_output_args(&mut cmd);
            }
            name = "CPU FFV1";
        } else {
            let encoder = choose_encoder(preference, &ffmpeg)?;
            encoder.raw_input(&mut cmd);
            cmd.arg("-an");
            encoder.output_args(&mut cmd);
            cmd.args(["-aspect", "4:3"]);
            name = encoder.name();
        }
        Encoder::colour_flags(&mut cmd);
        let mut child = EncodingChild(
            cmd.args(["-f", "matroska"])
                .arg(&partial)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|e| format!("FFmpeg: {e}"))?,
        );
        let pipe = child.0.stdin.take().ok_or("FFmpeg stdin unavailable")?;
        println!("DIRECT_MKV_START profile={profile} encoder={name} size={} tbc_files=false audio_mux=on_stop output={} writing={}", archive.as_ref().filter(|f| f.full).map_or("720x576".into(), |f| format!("{}x{}", f.width, f.height)), output.display(), partial.display());
        Ok(Self {
            base: base.into(),
            output,
            partial: Some(partial),
            capture,
            gpu,
            cpu: Cpu::new(),
            archive,
            components: vec![0.0; N * 3],
            viewing: vec![0; FRAME_BYTES],
            encoder: Some(child),
            pipe: Some(pipe),
            first_rf: None,
            retained_rf: Vec::new(),
            audio_offset,
            frames: 0,
            stats: Default::default(),
            first_field: None,
            burn,
        })
    }
    /// A teletext header image from the VBI receiver, for burning in.
    pub(crate) fn header(&mut self, image: &crate::vbi::HeaderImage) -> Result<(), String> {
        let Some(burner) = &mut self.burn else {
            return Ok(());
        };
        burner.add_header(image.clone());
        self.flush_burned(false, std::time::Duration::ZERO)
    }
    /// Encode the delayed frames whose header is now known (all when finishing).
    fn flush_burned(
        &mut self,
        finishing: bool,
        rendered: std::time::Duration,
    ) -> Result<(), String> {
        let Some(burner) = &mut self.burn else {
            return Ok(());
        };
        for frame in burner.ready(finishing) {
            write_frame(
                &mut self.pipe,
                &self.stats,
                &mut self.frames,
                &frame,
                rendered,
            )?;
        }
        Ok(())
    }
    pub(crate) fn field(
        &mut self,
        samples: &[u16],
        first: bool,
        rf: f64,
        field: u64,
    ) -> Result<(), String> {
        let parity = i32::from(!first);
        let rendering = Instant::now();
        let archive = self.archive.is_some();
        let bytes = if let Some(frame) = &mut self.archive {
            use crate::preview::RasterChroma;
            // Composite full raster carries no decoded colour.
            let mode = frame.full.then_some(frame.mode);
            if mode == Some(RasterChroma::Composite) {
            } else if let Some(gpu) = &self.gpu {
                gpu.components(samples, &mut self.components)?;
            } else {
                if mode == Some(RasterChroma::Whole) {
                    self.cpu.decode_full(samples);
                } else {
                    self.cpu.decode(
                        samples,
                        self.capture.start,
                        self.capture.end,
                        self.capture.burst_start,
                        self.capture.burst_end,
                    );
                }
                self.components.copy_from_slice(&self.cpu.c);
            }
            if first {
                self.first_rf = Some(rf);
                self.first_field = Some(field);
            }
            let complete = if frame.full {
                frame.field_full(&self.components, samples, &self.capture, parity)
            } else {
                frame.field(&self.components, &self.capture, parity)
            };
            if !complete {
                self.stats
                    .render_ns
                    .fetch_add(rendering.elapsed().as_nanos() as u64, Ordering::Relaxed);
                return Ok(());
            }
            self.retained_rf
                .extend([self.first_rf.take().ok_or("Missing first MKV field")?, rf]);
            &frame.data
        } else {
            if let Some(gpu) = &self.gpu {
                gpu.field(samples, parity, &mut self.viewing)?;
            } else {
                self.cpu
                    .field(samples, &self.capture, parity, &mut self.viewing);
            }
            self.retained_rf.push(rf);
            &self.viewing
        };
        let rendered = rendering.elapsed();
        // An archive frame is timed by its first field, a viewing frame by its field.
        let frame_field = if archive {
            self.first_field.unwrap_or(field)
        } else {
            field
        };
        if let Some(burner) = &mut self.burn {
            burner.push_frame(frame_field, bytes.to_vec());
            return self.flush_burned(false, rendered);
        }
        write_frame(
            &mut self.pipe,
            &self.stats,
            &mut self.frames,
            bytes,
            rendered,
        )
    }
    /// Give the finished or abandoned recording its final, visible name.
    fn reveal(&mut self) -> Result<(), String> {
        let Some(partial) = self.partial.take() else {
            return Ok(());
        };
        if !partial.exists() {
            return Ok(());
        }
        fs::rename(&partial, &self.output).map_err(|e| {
            format!(
                "Cannot rename {} to {}: {e}",
                partial.display(),
                self.output.display()
            )
        })
    }
    pub(crate) fn finish(&mut self, audio: bool) -> Result<(), String> {
        self.flush_burned(true, std::time::Duration::ZERO)?;
        drop(self.pipe.take());
        if let Some(mut child) = self.encoder.take() {
            let status = child.0.wait().map_err(|e| e.to_string());
            self.reveal()?;
            let status = status?;
            if !status.success() {
                return Err(format!(
                    "Direct MKV encoder failed ({status}); partial file retained: {}",
                    self.output.display()
                ));
            }
        }
        if self.frames == 0 {
            return Err("No complete video frame was recorded; MKV output is empty. Check RF sync or allow capture to run longer.".into());
        }
        let track = if audio && !self.retained_rf.is_empty() {
            // Stop closes capture, not the final audio alignment/mux operation.
            crate::audio::export_track(
                &self.base.to_string_lossy(),
                &self.output,
                &self.retained_rf,
                &(0..self.retained_rf.len()).collect::<Vec<_>>(),
                self.audio_offset,
                false,
            )?
        } else {
            None
        };
        if let Some(track) = track {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_nanos();
            let temporary = self
                .output
                .with_extension(format!("audio-mux-{}-{stamp}.mkv", std::process::id()));
            println!("DIRECT_MKV_AUDIO_MUX video_copy=true");
            let ffmpeg = std::env::var_os("VHS_FFMPEG").unwrap_or_else(|| "ffmpeg".into());
            let mut cmd = Command::new(ffmpeg);
            cmd.args([
                "-hide_banner",
                "-loglevel",
                "warning",
                "-nostdin",
                "-n",
                "-i",
            ])
            .arg(&self.output);
            crate::audio::ExportTrack::configure(Some(&track), &mut cmd);
            let status = cmd
                .args(["-c:v", "copy", "-f", "matroska"])
                .arg(&temporary)
                .status()
                .map_err(|e| e.to_string())?;
            if !status.success() {
                let _ = fs::remove_file(&temporary);
                return Err("Direct MKV audio mux failed; original video and WAV retained".into());
            }
            fs::rename(&temporary, &self.output).map_err(|e| {
                format!(
                    "Cannot replace MKV after audio mux: {e}; muxed file retained at {}",
                    temporary.display()
                )
            })?;
        }
        println!(
            "DIRECT_MKV_OK frames={} audio={} tbc_files=false output={}",
            self.frames,
            audio,
            self.output.display()
        );
        Ok(())
    }
}

impl Drop for DirectMkv {
    fn drop(&mut self) {
        // An unfinished capture still keeps whatever FFmpeg wrote, under the
        // visible name, instead of a hidden partial file.
        drop(self.pipe.take());
        drop(self.encoder.take());
        if let Err(e) = self.reveal() {
            eprintln!("Direct MKV: {e}");
        }
    }
}
fn partial_path(output: &Path) -> Result<PathBuf, String> {
    let name = output
        .file_name()
        .ok_or("MKV output needs a file name")?
        .to_string_lossy();
    Ok(output.with_file_name(format!(".{name}.partial")))
}

/// Hand one encoded frame to FFmpeg and account for it.
fn write_frame(
    pipe: &mut Option<std::process::ChildStdin>,
    stats: &DirectStats,
    frames: &mut usize,
    bytes: &[u8],
    rendered: std::time::Duration,
) -> Result<(), String> {
    let writing = Instant::now();
    pipe.as_mut()
        .ok_or("MKV encoder already finalized")?
        .write_all(bytes)
        .map_err(|e| format!("Direct MKV encoder write failed: {e}. Partial MKV retained."))?;
    stats
        .render_ns
        .fetch_add(rendered.as_nanos() as u64, Ordering::Relaxed);
    stats
        .write_ns
        .fetch_add(writing.elapsed().as_nanos() as u64, Ordering::Relaxed);
    *frames += 1;
    if frames.is_multiple_of(25) {
        println!("DIRECT_MKV_PROGRESS frames={frames} tbc_files=false");
    }
    Ok(())
}

enum DirectMessage {
    Field(Vec<u16>, bool, f64, u64),
    Header(crate::vbi::HeaderImage),
    Finish(bool),
}
// About 0.3 s of decoded fields: smooths encoder jitter, while sustained
// backpressure still reaches the bounded RF acquisition buffer.
const DIRECT_QUEUE: usize = 16;

/// Runs colour rendering and FFmpeg encoding on their own thread so they overlap
/// with RF demodulation and TBC on the decode thread. The encoder, including its
/// OpenCL context, is created, used and finalized only on that thread.
pub(crate) struct DirectMkvWorker {
    sender: Option<mpsc::SyncSender<DirectMessage>>,
    worker: Option<JoinHandle<Result<(), String>>>,
    queued: Arc<AtomicUsize>,
    stats: Arc<DirectStats>,
    encoder_pid: u32,
}
impl DirectMkvWorker {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        base: &Path,
        profile: &str,
        preference: &str,
        gpu: bool,
        device: usize,
        audio_offset: f64,
        full_raster: Option<crate::preview::RasterChroma>,
        burn_header: Option<crate::preview::HeaderPlacement>,
    ) -> Result<Self, String> {
        let (base, profile, preference) =
            (base.to_owned(), profile.to_owned(), preference.to_owned());
        let (sender, receiver) = mpsc::sync_channel(DIRECT_QUEUE);
        let (ready, started) = mpsc::channel();
        let queued = Arc::new(AtomicUsize::new(0));
        let worker_queued = queued.clone();
        let worker = std::thread::Builder::new()
            .name("direct-mkv".into())
            .spawn(move || {
                let mut direct = match DirectMkv::new(
                    &base,
                    &profile,
                    &preference,
                    gpu,
                    device,
                    audio_offset,
                    full_raster,
                    burn_header,
                ) {
                    Ok(direct) => {
                        let pid = direct.encoder.as_ref().map_or(0, |child| child.0.id());
                        let _ = ready.send(Ok((direct.stats.clone(), pid)));
                        direct
                    }
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return Ok(());
                    }
                };
                while let Ok(message) = receiver.recv() {
                    match message {
                        DirectMessage::Field(samples, first, rf, field) => {
                            worker_queued.fetch_sub(1, Ordering::AcqRel);
                            direct.field(&samples, first, rf, field)?;
                        }
                        DirectMessage::Header(image) => direct.header(&image)?,
                        DirectMessage::Finish(audio) => return direct.finish(audio),
                    }
                }
                // The decoder went away without finishing; the encoder is killed on drop.
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        let (stats, encoder_pid) = started
            .recv()
            .map_err(|_| "Direct MKV worker exited during startup".to_string())??;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
            queued,
            stats,
            encoder_pid,
        })
    }
    pub(crate) fn stats(&self) -> &DirectStats {
        &self.stats
    }
    /// FFmpeg process id, for live CPU diagnostics.
    pub(crate) fn encoder_pid(&self) -> u32 {
        self.encoder_pid
    }
    /// Fields waiting for the encoder thread.
    pub(crate) fn queued(&self) -> usize {
        self.queued.load(Ordering::Acquire)
    }
    fn join(&mut self) -> Result<(), String> {
        self.sender = None;
        match self.worker.take().map(JoinHandle::join) {
            Some(Ok(result)) => result,
            Some(Err(_)) => Err("Direct MKV worker panicked; partial MKV retained".into()),
            None => Err("MKV encoder already finalized".into()),
        }
    }
    fn send(&mut self, message: DirectMessage) -> Result<(), String> {
        let sent = self
            .sender
            .as_ref()
            .ok_or("MKV encoder already finalized")?
            .send(message);
        match sent {
            Ok(()) => Ok(()),
            // The worker only hangs up after an error; report that error.
            Err(_) => self
                .join()
                .and(Err("Direct MKV worker stopped unexpectedly".into())),
        }
    }
    pub(crate) fn field(
        &mut self,
        samples: &[u16],
        first: bool,
        rf: f64,
        field: u64,
    ) -> Result<(), String> {
        self.queued.fetch_add(1, Ordering::AcqRel);
        let sending = Instant::now();
        let result = self.send(DirectMessage::Field(samples.to_vec(), first, rf, field));
        self.stats
            .send_ns
            .fetch_add(sending.elapsed().as_nanos() as u64, Ordering::Relaxed);
        result
    }
    pub(crate) fn header(&mut self, image: crate::vbi::HeaderImage) -> Result<(), String> {
        self.send(DirectMessage::Header(image))
    }
    pub(crate) fn finish(&mut self, audio: bool) -> Result<(), String> {
        self.send(DirectMessage::Finish(audio))?;
        self.join()
    }
}
impl Drop for DirectMkvWorker {
    fn drop(&mut self) {
        self.sender = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn archive_preserves_field_lines_precision_padding_and_pair_order() {
        let (_, c) = synthetic();
        let mut f = ArchiveFrame::new(&c);
        assert_eq!(f.width, 928);
        let mut components = vec![0.0; N * 3];
        for row in 22..310 {
            components[row * W..(row + 1) * W].fill(c.black + (row - 22) as f32 * c.range / 876.0);
        }
        assert!(
            !f.field(&components, &c, 1),
            "Leading second field must be skipped"
        );
        assert!(!f.field(&components, &c, 0));
        // The second field has a different exposure; lines must never be mixed.
        for value in &mut components[..N] {
            *value += c.range / 2.0;
        }
        assert!(f.field(&components, &c, 1));
        let word = |index| u16::from_le_bytes([f.data[index * 2], f.data[index * 2 + 1]]);
        assert_eq!(word(0), 64, "Padding must remain black");
        assert_eq!(word(3), 64);
        assert_eq!(word(928 + 3), 502);
        assert_eq!(word(2 * 928 + 3), 65, "Single 10-bit steps must survive");
        assert_eq!(word(575 * 928 + 3), 789);
        assert_eq!(word(928 * OH), 512, "Neutral chroma must stay neutral");
        assert!(
            !f.field(&components, &c, 1),
            "Second field cannot reuse an old first field"
        );
    }
    fn synthetic() -> (Vec<u16>, Capture) {
        let null = if cfg!(windows) { "NUL" } else { "/dev/null" };
        let file = || BufReader::new(File::open(null).unwrap());
        let c = Capture {
            luma: Some(file()),
            chroma: Some(file()),
            parity: vec![0, 1],
            rf_fields: vec![0.0, 0.0],
            start: 185,
            end: 1107,
            burst_start: 98,
            burst_end: 138,
            black: 16384.0,
            range: 37632.0,
        };
        let mut input = vec![35199; N * 2];
        for row in 0..H {
            for x in 0..W {
                // Independently encode known U=4000, V=-2500 PAL components.
                // The burst points towards -U with alternating +/-V swing.
                let angle = (x as f64 - row as f64) * std::f64::consts::FRAC_PI_2;
                let vs = if row % 2 == 0 { -1.0 } else { 1.0 };
                let burst = 5000.0 * (-angle.cos() + vs * angle.sin());
                let value = if (98..138).contains(&x) {
                    burst
                } else if x >= 185 {
                    4000.0 * angle.cos() + vs * (-2500.0) * angle.sin()
                } else {
                    0.0
                };
                input[N + row * W + x] = (32767.0 + value).round() as u16;
            }
        }
        (input, c)
    }
    #[test]
    fn native_colour_filter_preserves_neutral_luma_and_saturated_chroma() {
        let (mut input, c) = synthetic();
        let mut cpu = Cpu::new();
        let mut out = vec![0; FRAME_BYTES];
        cpu.field(&input, &c, 0, &mut out);
        assert!((out[100 * OW + 300] as i32 - 125).abs() <= 1);
        for row in [50, 51, 100, 101] {
            let at = OW * OH + row * OW + 300;
            assert!(
                (out[at] as i32 - 155).abs() <= 1,
                "Incorrect Cb or alternating PAL hue"
            );
            assert!(
                (out[at + 1] as i32 - 116).abs() <= 1,
                "Incorrect Cr or alternating PAL hue"
            );
        }
        input[N..].fill(32767);
        cpu.field(&input, &c, 1, &mut out);
        assert!(out[OW * OH..].iter().all(|v| *v == 128));
    }
    #[test]
    #[ignore = "requires a working OpenCL GPU"]
    fn opencl_export_matches_cpu_for_both_field_parities() {
        let (input, c) = synthetic();
        let gpu = Gpu::new(&c, 0).unwrap();
        let mut cpu = Cpu::new();
        for parity in [0, 1] {
            let mut a = vec![0; FRAME_BYTES];
            let mut b = a.clone();
            cpu.field(&input, &c, parity, &mut a);
            gpu.field(&input, parity, &mut b).unwrap();
            let max = a.iter().zip(&b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
            assert!(max <= 1, "CPU/GPU maximum error: {max}");
        }
        let mut components = vec![0.0; N * 3];
        gpu.components(&input, &mut components).unwrap();
        let mut a = ArchiveFrame::new(&c);
        let mut b = ArchiveFrame::new(&c);
        for parity in [0, 1] {
            assert_eq!(
                a.field(&cpu.c, &c, parity),
                b.field(&components, &c, parity)
            );
        }
        let max = a
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .zip(b.data.as_chunks::<2>().0)
            .map(|(x, y)| u16::from_le_bytes(*x).abs_diff(u16::from_le_bytes(*y)))
            .max()
            .unwrap();
        assert!(max <= 1, "Archive CPU/GPU maximum 10-bit error: {max}");
    }
    #[test]
    fn colour_preview_preserves_neutral_levels_and_known_chroma() {
        let (mut input, _) = synthetic();
        let mut cpu = Cpu::new();
        let colour = cpu.preview(&input);
        assert!(colour[OW * 288..]
            .as_chunks::<2>()
            .0
            .iter()
            .any(|uv| uv[0] > 145 && uv[1] < 125));
        input[N..].fill(32767);
        for (level, expected) in [(crate::preview::BLACK, 16), (crate::preview::WHITE, 235)] {
            input[..N].fill(level as u16);
            let neutral = cpu.preview(&input);
            assert!(neutral[..OW * 288].iter().all(|&v| v == expected));
            assert!(neutral[OW * 288..].iter().all(|&v| v == 128));
        }
    }
    #[test]
    #[ignore = "requires a working OpenCL GPU"]
    fn opencl_colour_preview_matches_cpu_without_field_interpolation() {
        let (input, c) = synthetic();
        let gpu = Gpu::new(&c, 0).unwrap();
        gpu.input.write(&input).enq().unwrap();
        let preview = GpuPreview::new(&gpu._pq, &gpu.input, None).unwrap();
        let actual = preview.render().unwrap();
        let expected = Cpu::new().preview(&input);
        assert_eq!(actual.len(), OW * 288 * 3 / 2);
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(max <= 1, "CPU/GPU colour preview maximum error: {max}");
    }
}
