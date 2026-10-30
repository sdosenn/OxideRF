//! Batched OpenCL demodulation and TBC with a persistent device-side signal ring.
use crate::native::{
    lowpass, pal_rf_gain, pal_sync_gain, pal_video_gain, Rf, C, EDGE, HEIGHT, HOP, N, UNDER, WIDTH,
};
use ocl::{Buffer, Kernel, ProQue};

const MAX_BATCH: usize = 64;

// Nonblocking uploads borrow host slices. On an early return or unwind, drain
// the queue before those slices can be dropped. Successful blocking reads below
// already establish completion and disarm this guard.
struct UploadFence {
    queue: ocl::Queue,
    pending: bool,
}
impl Drop for UploadFence {
    fn drop(&mut self) {
        if self.pending {
            let _ = self.queue.finish();
        }
    }
}

/// Result of [`Pipeline::refine_field`].
pub(crate) struct RefinedField {
    /// First-pass tip and porch levels of each line still locked after it.
    pub levels: Vec<Option<(f32, f32)>>,
}

/// Host line data borrowed by a field queued with [`Pipeline::begin_field`].
pub(crate) struct FieldUpload {
    // Declared first: dropping drains the queue before the data is freed.
    fence: UploadFence,
    _origins: Vec<u32>,
    _lines: Vec<f32>,
    /// Envelope length and threshold fraction when dropouts are detected
    /// with the field.
    dropouts: Option<(usize, f32)>,
}

/// Dropout detection kept on the GPU: envelope smoothing, the field's RF
/// level and the interval boundaries, read back with the field.
struct DropoutGpu {
    forward: Kernel,
    backward: Kernel,
    median: Kernel,
    chunks: Kernel,
    entries: Kernel,
    events: Kernel,
    smoothed: Buffer<f32>,
    level: Buffer<f32>,
    event_list: Buffer<u32>,
    event_count: Buffer<u32>,
    _forward_out: Buffer<f32>,
    _state: Buffer<u32>,
    _entry: Buffer<u32>,
}
/// Raw dropout intervals as sample ranges of a field envelope.
type Intervals = Vec<(usize, usize)>;
/// Boundary events the GPU stores per field; fields with more fall back to
/// detection on the host.
const DROPOUT_EVENTS: usize = 65536;
/// Events read back with every field; more need a second read.
const DROPOUT_EVENTS_READ: usize = 4096;
const DROPOUT_CHUNK: usize = 256;

pub(crate) struct Pipeline {
    _pq: ProQue,
    mask: u32,
    batch: usize,
    ring: Buffer<f32>,
    sinc_lut: Buffer<f32>,
    raw: RfBuffers,
    phases: Buffer<f32>,
    count: Buffer<u32>,
    histogram: Buffer<u32>,
    slice: Buffer<f32>,
    histogram_kernel: Kernel,
    choose_slice: Kernel,
    events: Buffer<u64>,
    event_capacity: usize,
    origins: Buffer<u32>,
    lines: Buffer<f32>,
    refined: Buffer<f32>,
    refined_at: Buffer<f32>,
    threshold: Buffer<f32>,
    output: Buffer<u16>,
    demod: Vec<(Kernel, usize)>,
    store: Kernel,
    pulses: Kernel,
    refine: Kernel,
    refine_threshold: Kernel,
    refine_at: Kernel,
    burst: Kernel,
    tbc: Kernel,
    chroma_state: Kernel,
    smooth_gain: Kernel,
    gain_stats: Kernel,
    _pal_state: Buffer<i32>,
    fs: f64,
    monitor: Option<(Kernel, Buffer<u8>)>,
    vbi: Option<(Kernel, Buffer<u16>)>,
    colour_preview: Option<crate::export::GpuPreview>,
    envelope: Buffer<f32>,
    dropout_repair: Option<(Buffer<u8>, Buffer<u16>, Kernel)>,
    dropout_gpu: Option<DropoutGpu>,
    // Keep all backing allocations alive for cached kernel arguments.
    _work: Vec<Buffer<f32>>,
}

/// RF upload buffers and kernels for each capture sample format.
struct RfBuffers {
    f32: (Buffer<f32>, Kernel, Kernel),
    u8: (Buffer<u8>, Kernel, Kernel),
    u16: (Buffer<u16>, Kernel, Kernel),
    _means: Buffer<f32>,
}
impl RfBuffers {
    fn new(
        pq: &ProQue,
        len: usize,
        means: &Buffer<f32>,
        packed: &Buffer<f32>,
    ) -> ocl::Result<Self> {
        fn input<T: ocl::OclPrm>(
            pq: &ProQue,
            name: &str,
            len: usize,
            means: &Buffer<f32>,
            packed: &Buffer<f32>,
        ) -> ocl::Result<(Buffer<T>, Kernel, Kernel)> {
            let raw = buffer::<T>(pq, len)?;
            let block_means = pq
                .kernel_builder(format!("block_means_{name}"))
                .global_work_size(256)
                .local_work_size(256)
                .arg(&raw)
                .arg(means)
                .arg_local::<f32>(256)
                .build()?;
            let pack = pq
                .kernel_builder(format!("pack_rf_{name}"))
                .global_work_size([N, 1])
                .arg(&raw)
                .arg(means)
                .arg(packed)
                .build()?;
            Ok((raw, block_means, pack))
        }
        Ok(Self {
            f32: input(pq, "f32", len, means, packed)?,
            u8: input(pq, "u8", len, means, packed)?,
            u16: input(pq, "u16", len, means, packed)?,
            _means: means.clone(),
        })
    }
    /// Queue a non-blocking upload of the first `len` samples; returns the
    /// mean and pack kernels for that format.
    ///
    /// # Safety
    /// `raw` must outlive the upload, e.g. until a later blocking read.
    unsafe fn upload(&self, raw: &Rf, len: usize) -> ocl::Result<(&Kernel, &Kernel)> {
        let (means, pack) = match raw {
            Rf::F32(v) => {
                self.f32.0.write(&v[..len]).block(false).enq()?;
                (&self.f32.1, &self.f32.2)
            }
            Rf::U8(v) => {
                self.u8.0.write(&v[..len]).block(false).enq()?;
                (&self.u8.1, &self.u8.2)
            }
            Rf::U16(v) => {
                self.u16.0.write(&v[..len]).block(false).enq()?;
                (&self.u16.1, &self.u16.2)
            }
        };
        Ok((means, pack))
    }
}
fn buffer<T: ocl::OclPrm>(pq: &ProQue, len: usize) -> ocl::Result<Buffer<T>> {
    Buffer::builder().queue(pq.queue().clone()).len(len).build()
}
fn upload(pq: &ProQue, data: &[f32]) -> ocl::Result<Buffer<f32>> {
    Buffer::builder()
        .queue(pq.queue().clone())
        .len(data.len())
        .copy_host_slice(data)
        .build()
}
fn fft(
    pq: &ProQue,
    src: &Buffer<f32>,
    dst: &Buffer<f32>,
    twiddle: &Buffer<f32>,
    inverse: i32,
    batch: usize,
) -> ocl::Result<Vec<(Kernel, usize)>> {
    Ok(vec![
        (
            pq.kernel_builder("local_begin")
                .global_work_size([N / 2, batch])
                .local_work_size([256, 1])
                .arg(src)
                .arg(dst)
                .arg(twiddle)
                .arg(N.trailing_zeros())
                .arg(N as u32)
                .arg(inverse)
                .build()?,
            N / 2,
        ),
        (
            pq.kernel_builder("local_end")
                .global_work_size([N / 2, batch])
                .local_work_size([N / 1024, 1])
                .arg(dst)
                .arg(twiddle)
                .arg(N as u32)
                .arg(inverse)
                .build()?,
            N / 2,
        ),
    ])
}
impl Pipeline {
    pub(crate) fn new(fs: f64) -> Result<Self, String> {
        Self::build(fs, false, false).map_err(|e| e.to_string())
    }
    pub(crate) fn new_monitor(fs: f64) -> Result<Self, String> {
        Self::build(fs, true, true).map_err(|e| e.to_string())
    }
    pub(crate) fn new_colour_monitor(fs: f64) -> Result<Self, String> {
        Self::build(fs, false, true).map_err(|e| e.to_string())
    }
    fn build(fs: f64, luma_only: bool, monitor: bool) -> ocl::Result<Self> {
        // Previews use smaller batches for lower latency; below 32 the per-batch
        // overhead keeps an Apple M1 under real time with dropout correction.
        let batch = std::env::var("VHS_GPU_BATCH")
            .unwrap_or_else(|_| if monitor { "32" } else { "64" }.into())
            .parse::<usize>()
            .map_err(|_| ocl::Error::from("VHS_GPU_BATCH must be an integer in 1..=64"))?;
        if !(1..=MAX_BATCH).contains(&batch) {
            return Err("VHS_GPU_BATCH must be in 1..=64".into());
        }
        let engine = crate::Engine::new().map_err(ocl::Error::from)?;
        let pq = engine.pq;
        let capacity = ((fs * 0.075) as usize + batch * HOP + N).next_power_of_two();
        let mask = (capacity - 1) as u32;
        let ring = buffer::<f32>(&pq, capacity * 4)?;
        let raw_len = N + (batch - 1) * HOP;
        let means = buffer::<f32>(&pq, batch)?;
        let phases = buffer(&pq, batch * 2)?;
        let event_capacity = (batch * HOP) / (fs * 3e-6).floor().max(1.0) as usize + 32;
        let count = buffer(&pq, 1)?;
        let histogram = buffer::<u32>(&pq, 128)?;
        let slice = buffer::<f32>(&pq, 1)?;
        let events = buffer::<u64>(&pq, event_capacity * 2)?;
        let origins = buffer::<u32>(&pq, HEIGHT + 1)?;
        let lines = buffer::<f32>(&pq, (HEIGHT + 1) * 4)?;
        let sinc_lut = upload(&pq, crate::resample::table())?;
        let refined = buffer::<f32>(&pq, (HEIGHT + 1) * 4)?;
        let refined_at = buffer::<f32>(&pq, (HEIGHT + 1) * 4)?;
        let threshold = buffer::<f32>(&pq, 3)?;
        let row_burst = buffer::<f32>(&pq, HEIGHT * 4)?;
        let output = buffer::<u16>(&pq, WIDTH * HEIGHT * 2)?;
        let mut tw = Vec::with_capacity(N);
        let mut oscillator = Vec::with_capacity(2 * N);
        let mut rf_gain = Vec::with_capacity(N);
        let mut chroma_gain = Vec::with_capacity(N);
        let mut video_gain = Vec::with_capacity(2 * N);
        let mut sync_gain = Vec::with_capacity(N);
        for k in 0..N {
            let f = k as f64 * fs / N as f64;
            if k < N / 2 {
                let w = C::polar(-std::f64::consts::TAU * k as f64 / N as f64);
                tw.extend([w.re, w.im]);
            }
            let rotation = C::polar(-std::f64::consts::TAU * UNDER * k as f64 / fs);
            oscillator.extend([rotation.re, rotation.im]);
            rf_gain.push(if k < N / 2 { pal_rf_gain(f, fs) } else { 0.0 });
            chroma_gain.push(if k < N / 2 {
                2.0 * (1.0 - lowpass(f, 0.15e6, 0.1e6)) * lowpass(f, 1.0e6, 0.15e6)
            } else {
                0.0
            });
            let signed = if k <= N / 2 {
                k as f64
            } else {
                k as f64 - N as f64
            };
            let frequency = signed * fs / N as f64;
            let gain = pal_video_gain(frequency, fs);
            video_gain.extend([gain.re, gain.im]);
            sync_gain.push(pal_sync_gain(k.min(N - k) as f64 * fs / N as f64));
        }
        let tw = upload(&pq, &tw)?;
        let oscillator = upload(&pq, &oscillator)?;
        let rf_gain = upload(&pq, &rf_gain)?;
        let chroma_gain = upload(&pq, &chroma_gain)?;
        let video_gain = upload(&pq, &video_gain)?;
        let sync_gain = upload(&pq, &sync_gain)?;
        let mut work = Vec::new();
        for index in 0..12 {
            work.push(buffer::<f32>(
                &pq,
                if luma_only && index >= 10 {
                    2
                } else {
                    N * batch * 2
                },
            )?);
        }
        let raw = RfBuffers::new(&pq, raw_len, &means, &work[0])?;
        let mut demod = fft(&pq, &work[0], &work[1], &tw, 0, batch)?;
        demod.push((
            pq.kernel_builder("rf_bands")
                .global_work_size([N, batch])
                .arg(&work[1])
                .arg(&rf_gain)
                .arg(&chroma_gain)
                .arg(&work[2])
                .arg(&work[10])
                .arg(i32::from(luma_only))
                .build()?,
            N,
        ));
        demod.extend(fft(&pq, &work[2], &work[3], &tw, 1, batch)?);
        demod.push((
            pq.kernel_builder("fm_phase")
                .global_work_size([N, batch])
                .arg(&work[3])
                .arg(&work[4])
                .arg((fs / std::f64::consts::TAU) as f32)
                .build()?,
            N,
        ));
        demod.extend(fft(&pq, &work[4], &work[5], &tw, 0, batch)?);
        demod.push((
            pq.kernel_builder("video_bands")
                .global_work_size([N, batch])
                .arg(&work[5])
                .arg(&video_gain)
                .arg(&sync_gain)
                .arg(&work[6])
                .arg(&work[8])
                .build()?,
            N,
        ));
        demod.extend(fft(&pq, &work[6], &work[7], &tw, 1, batch)?);
        demod.extend(fft(&pq, &work[8], &work[9], &tw, 1, batch)?);
        if !luma_only {
            demod.extend(fft(&pq, &work[10], &work[11], &tw, 1, batch)?);
        }
        let envelope = buffer::<f32>(&pq, 1)?;
        let store = pq
            .kernel_builder("store_signal")
            .global_work_size([HOP, batch])
            .arg(&work[7])
            .arg(&work[9])
            .arg(&work[11])
            .arg(&oscillator)
            .arg(&phases)
            .arg(&ring)
            .arg(0u64)
            .arg(mask)
            .arg(i32::from(luma_only))
            .arg(&work[3])
            .arg(&envelope)
            .arg(0i32)
            .build()?;
        let histogram_kernel = pq
            .kernel_builder("sync_histogram")
            .global_work_size(HOP * batch)
            .local_work_size(256)
            .arg(&ring)
            .arg(mask)
            .arg(0u64)
            .arg(&histogram)
            .build()?;
        let choose_slice = pq
            .kernel_builder("choose_sync_slice")
            .global_work_size(1)
            .arg(&histogram)
            .arg((HOP * batch) as u32)
            .arg(&slice)
            .build()?;
        let pulses = pq
            .kernel_builder("find_pulses")
            .global_work_size(HOP * batch)
            .arg(&ring)
            .arg(mask)
            .arg(0u64)
            .arg(EDGE as u64)
            .arg((fs / 1e6) as f32)
            .arg(&count)
            .arg(&events)
            .arg(event_capacity as u32)
            .arg(&slice)
            .build()?;
        let refine = pq
            .kernel_builder("refine_lines")
            .global_work_size(HEIGHT + 1)
            .arg(&ring)
            .arg(mask)
            .arg(&origins)
            .arg(&lines)
            .arg((fs / 1e6) as f32)
            .arg(&refined)
            .arg(0.0f32)
            .build()?;
        let refine_threshold = pq
            .kernel_builder("refine_threshold")
            .global_work_size(278)
            .arg(&refined)
            .arg(&threshold)
            .build()?;
        let refine_at = pq
            .kernel_builder("refine_lines_at")
            .global_work_size(HEIGHT + 1)
            .arg(&ring)
            .arg(mask)
            .arg(&origins)
            .arg(&refined)
            .arg((fs / 1e6) as f32)
            .arg(&refined_at)
            .arg(&threshold)
            .build()?;
        let burst = pq
            .kernel_builder("burst_rows")
            .global_work_size(HEIGHT)
            .arg(&ring)
            .arg(mask)
            .arg(&origins)
            .arg(&lines)
            .arg(&row_burst)
            .build()?;
        let pal_state = buffer::<i32>(&pq, 1)?;
        let chroma_state = pq
            .kernel_builder("pal_chroma_state")
            .global_work_size(256)
            .local_work_size(256)
            .arg(&row_burst)
            .arg(&pal_state)
            .arg_local::<f32>(3 * 256)
            .build()?;
        let gain_cap = buffer::<f32>(&pq, 1)?;
        let smoothed_gain = buffer::<f32>(&pq, HEIGHT)?;
        let gain_stats = pq
            .kernel_builder("burst_gain_stats")
            .global_work_size(256)
            .local_work_size(256)
            .arg(&row_burst)
            .arg(&gain_cap)
            .build()?;
        let smooth_gain = pq
            .kernel_builder("smooth_burst_gain")
            .global_work_size(HEIGHT)
            .arg(&row_burst)
            .arg(&gain_cap)
            .arg(&smoothed_gain)
            .build()?;
        let tbc = pq
            .kernel_builder("tbc_field")
            .global_work_size(WIDTH * HEIGHT)
            .arg(&ring)
            .arg(mask)
            .arg(&origins)
            .arg(&lines)
            .arg(&row_burst)
            .arg(4.1e6f32)
            .arg(&output)
            .arg(&pal_state)
            .arg(&smoothed_gain)
            .arg(&sinc_lut)
            .build()?;
        let monitor = if luma_only {
            let pixels = buffer::<u8>(&pq, crate::preview::WIDTH * crate::preview::HEIGHT)?;
            let kernel = pq
                .kernel_builder("preview_luma")
                .global_work_size([crate::preview::WIDTH, crate::preview::HEIGHT])
                .arg(&ring)
                .arg(mask)
                .arg(&origins)
                .arg(&lines)
                .arg(4.1e6f32)
                .arg(&pixels)
                .arg(&sinc_lut)
                .build()?;
            Some((kernel, pixels))
        } else {
            None
        };

        work.extend([
            tw,
            oscillator,
            rf_gain,
            chroma_gain,
            video_gain,
            sync_gain,
            row_burst,
            gain_cap,
            smoothed_gain,
        ]);
        eprintln!(
            "OpenCL resident pipeline: {batch} RF blocks/batch, {} MiB signal ring",
            capacity * 16 / (1 << 20)
        );
        Ok(Self {
            _pq: pq,
            mask,
            batch,
            ring,
            sinc_lut,
            raw,
            phases,
            count,
            histogram,
            slice,
            histogram_kernel,
            choose_slice,
            events,
            event_capacity,
            origins,
            lines,
            refined,
            refined_at,
            threshold,
            output,
            demod,
            store,
            pulses,
            refine,
            refine_threshold,
            refine_at,
            burst,
            tbc,
            chroma_state,
            smooth_gain,
            gain_stats,
            _pal_state: pal_state,
            fs,
            monitor,
            vbi: None,
            colour_preview: None,
            envelope,
            dropout_repair: None,
            dropout_gpu: None,
            _work: work,
        })
    }
    pub(crate) fn capacity(&self) -> usize {
        self.ring.len() / 4
    }
    pub(crate) fn enable_dropouts(&mut self) -> Result<(), String> {
        let run = || {
            let envelope = buffer::<f32>(&self._pq, self.capacity())?;
            let mask = buffer::<u8>(&self._pq, WIDTH * HEIGHT)?;
            mask.cmd().fill(0u8, None).enq()?;
            let output = buffer::<u16>(&self._pq, WIDTH * HEIGHT * 2)?;
            let kernel = self
                ._pq
                .kernel_builder("repair_dropouts")
                .global_work_size(WIDTH * HEIGHT * 2)
                .arg(&self.output)
                .arg(&mask)
                .arg(&output)
                .build()?;
            self.store.set_arg(10, &envelope)?;
            self.store.set_arg(11, 1i32)?;
            let (b, a) = crate::dropout::smoothing(self.fs);
            let length = self.smoothing_capacity();
            let chunks = length.div_ceil(DROPOUT_CHUNK);
            let forward_out = buffer::<f32>(&self._pq, length)?;
            let smoothed = buffer::<f32>(&self._pq, length)?;
            let level = buffer::<f32>(&self._pq, 2)?;
            let state = buffer::<u32>(&self._pq, chunks)?;
            let entry = buffer::<u32>(&self._pq, chunks)?;
            let event_list = buffer::<u32>(&self._pq, DROPOUT_EVENTS)?;
            let event_count = buffer::<u32>(&self._pq, 1)?;
            let pq = &self._pq;
            let gpu = DropoutGpu {
                forward: pq
                    .kernel_builder("smooth_forward")
                    .global_work_size(1)
                    .arg(&envelope)
                    .arg(self.mask)
                    .arg(0u32)
                    .arg(0u32)
                    .arg(b)
                    .arg(a)
                    .arg(&forward_out)
                    .build()?,
                backward: pq
                    .kernel_builder("smooth_backward")
                    .global_work_size(1)
                    .arg(&forward_out)
                    .arg(0u32)
                    .arg(b)
                    .arg(a)
                    .arg(&smoothed)
                    .build()?,
                median: pq
                    .kernel_builder("dropout_median")
                    .global_work_size(256)
                    .local_work_size(256)
                    .arg(&smoothed)
                    .arg(0u32)
                    .arg(&level)
                    .arg_local::<u32>(256)
                    .build()?,
                chunks: pq
                    .kernel_builder("dropout_chunks")
                    .global_work_size(1)
                    .arg(&smoothed)
                    .arg(0u32)
                    .arg(&level)
                    .arg(0f32)
                    .arg(&state)
                    .build()?,
                entries: pq
                    .kernel_builder("dropout_entries")
                    .global_work_size(256)
                    .local_work_size(256)
                    .arg(&state)
                    .arg(0u32)
                    .arg(&entry)
                    .arg_local::<u32>(256)
                    .build()?,
                events: pq
                    .kernel_builder("dropout_events")
                    .global_work_size(1)
                    .arg(&smoothed)
                    .arg(0u32)
                    .arg(&level)
                    .arg(0f32)
                    .arg(&entry)
                    .arg(&event_list)
                    .arg(&event_count)
                    .arg(DROPOUT_EVENTS as u32)
                    .build()?,
                smoothed,
                level,
                event_list,
                event_count,
                _forward_out: forward_out,
                _state: state,
                _entry: entry,
            };
            Ok::<_, ocl::Error>((envelope, mask, output, kernel, gpu))
        };
        let (envelope, mask, output, kernel, gpu) = run().map_err(|e| e.to_string())?;
        self.envelope = envelope;
        self.dropout_repair = Some((mask, output, kernel));
        self.dropout_gpu = Some(gpu);
        Ok(())
    }
    /// Longest field envelope the GPU smooths: a PAL field plus margin.
    pub(crate) fn smoothing_capacity(&self) -> usize {
        (self.fs * 0.025) as usize
    }
    pub(crate) fn dropout_map(
        &self,
        locations: &[f64],
        fraction: f32,
    ) -> Result<crate::dropout::Map, String> {
        let start = locations[0].floor() as u64;
        let count = (locations[HEIGHT].ceil() as u64 - start) as usize;
        if count > self.capacity() || self.dropout_repair.is_none() {
            return Err("Invalid dropout envelope range".into());
        }
        let at = (start as u32 & self.mask) as usize;
        let first = count.min(self.capacity() - at);
        let mut env = vec![0f32; count];
        self.envelope
            .read(&mut env[..first])
            .offset(at)
            .enq()
            .map_err(|e| e.to_string())?;
        if first < count {
            self.envelope
                .read(&mut env[first..])
                .offset(0)
                .enq()
                .map_err(|e| e.to_string())?;
        }
        Ok(crate::dropout::detect(
            &env, start, locations, self.fs, fraction,
        ))
    }
    pub(crate) fn set_dropout_mask(&self, map: &crate::dropout::Map) -> Result<(), String> {
        self.dropout_repair
            .as_ref()
            .ok_or("Dropout repair not configured")?
            .0
            .write(&map.mask)
            .enq()
            .map_err(|e| e.to_string())
    }
    #[cfg(test)]
    pub(crate) fn read_signal(&self, absolute: u64, count: usize) -> Vec<f32> {
        let offset = (absolute as usize & self.mask as usize) * 4;
        let first = (self.ring.len() - offset).min(count * 4);
        let mut values = vec![0f32; count * 4];
        self.ring
            .read(&mut values[..first])
            .offset(offset)
            .enq()
            .unwrap();
        if first < values.len() {
            self.ring.read(&mut values[first..]).enq().unwrap();
        }
        values
    }
    pub(crate) fn batch_samples(&self) -> usize {
        self.batch * HOP + 2 * EDGE
    }
    #[cfg(test)]
    pub(crate) fn demod(&mut self, raw: &[f32], absolute: u64) -> Result<Vec<(u64, u64)>, String> {
        self.demod_rf(&Rf::F32(raw.to_vec()), raw.len(), absolute)
    }
    /// Demodulate the first `len` samples of `raw`, in its capture format.
    pub(crate) fn demod_rf(
        &mut self,
        raw: &Rf,
        len: usize,
        absolute: u64,
    ) -> Result<Vec<(u64, u64)>, String> {
        self.demod_inner(raw, len, absolute)
            .map_err(|e| e.to_string())
    }
    fn demod_inner(&mut self, raw: &Rf, len: usize, absolute: u64) -> ocl::Result<Vec<(u64, u64)>> {
        let blocks = (len - 2 * EDGE) / HOP;
        let mut phases = Vec::with_capacity(blocks * 2);
        for b in 0..blocks {
            let phase = C::polar(
                (-std::f64::consts::TAU * UNDER * (absolute + b as u64 * HOP as u64) as f64
                    / self.fs)
                    .rem_euclid(std::f64::consts::TAU),
            );
            phases.extend([phase.re, phase.im]);
        }
        let zero = [0u32];
        let histogram = [0u32; 128];
        let mut uploads = UploadFence {
            queue: self._pq.queue().clone(),
            pending: true,
        };
        // All borrowed inputs outlive the completion reads and UploadFence.
        // Avoid five GPU/CPU round trips before each small live RF batch.
        let (means, pack) = unsafe { self.raw.upload(raw, len)? };
        unsafe {
            self.phases.write(&phases).block(false).enq()?;
            self.count.write(&zero[..]).block(false).enq()?;
            self.histogram.write(&histogram[..]).block(false).enq()?;
        }
        self.histogram_kernel.set_arg(2, absolute + EDGE as u64)?;
        self.choose_slice.set_arg(1, (HOP * blocks) as u32)?;
        self.store.set_arg(6, absolute)?;
        self.pulses.set_arg(2, absolute + EDGE as u64)?;
        // Sizes and kernel arguments are validated here; all work uses one in-order queue.
        unsafe {
            means.cmd().global_work_size(256 * blocks).enq()?;
            pack.cmd().global_work_size([N, blocks]).enq()?;
            for (kernel, width) in &self.demod {
                kernel.cmd().global_work_size([*width, blocks]).enq()?;
            }
            self.store.cmd().global_work_size([HOP, blocks]).enq()?;
            self.histogram_kernel
                .cmd()
                .global_work_size(HOP * blocks)
                .enq()?;
            self.choose_slice.enq()?;
            self.pulses.cmd().global_work_size(HOP * blocks).enq()?;
        }
        if std::env::var_os("VHS_NATIVE_STATS").is_some() {
            let mut level = [0f32];
            self.slice.read(&mut level[..]).enq()?;
            eprintln!(
                "SYNC_SLICE time_s={:.4} level_hz={:.0}",
                absolute as f64 / self.fs,
                level[0]
            );
        }
        let mut count = [0u32];
        self.count.read(&mut count[..]).enq()?;
        if count[0] as usize > self.event_capacity {
            return Err("OpenCL pulse event capacity exceeded".into());
        }
        let mut events = vec![0u64; count[0] as usize * 2];
        if !events.is_empty() {
            self.events.read(&mut events).enq()?;
        }
        let mut result: Vec<_> = events
            .as_chunks::<2>()
            .0
            .iter()
            .map(|e| (e[0], e[1]))
            .collect();
        result.sort_unstable_by_key(|e| e.0);
        uploads.pending = false;
        Ok(result)
    }
    #[cfg(test)]
    pub(crate) fn refine_at_level(
        &mut self,
        locations: &mut [f64],
        locked: &mut [bool],
        threshold: f32,
    ) -> Result<Vec<(f32, f32)>, String> {
        self.refine_inner(locations, locked, threshold)
            .map_err(|e| e.to_string())
    }
    #[cfg(test)]
    fn refine_inner(
        &mut self,
        locations: &mut [f64],
        locked: &mut [bool],
        threshold: f32,
    ) -> ocl::Result<Vec<(f32, f32)>> {
        self.refine.set_arg(6, threshold)?;
        let origins: Vec<_> = locations
            .iter()
            .map(|p| (*p as u64 as u32) & self.mask)
            .collect();
        let lines: Vec<_> = locations
            .iter()
            .zip(locked.iter())
            .flat_map(|(p, locked)| [p.fract() as f32, 0.0, if *locked { 1.0 } else { 0.0 }, 0.0])
            .collect();
        self.origins.write(&origins).enq()?;
        self.lines.write(&lines).enq()?;
        unsafe {
            self.refine.enq()?;
        }
        let mut result = vec![0f32; locations.len() * 4];
        self.refined.read(&mut result).enq()?;
        for (p, row) in locations.iter_mut().zip(result.as_chunks::<4>().0.iter()) {
            *p = p.floor() + row[0] as f64;
        }
        for (valid, row) in locked.iter_mut().zip(result.as_chunks::<4>().0.iter()) {
            *valid &= row[3] > 0.0;
        }
        Ok(result
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| (v[1], v[2]))
            .collect())
    }
    /// Both sync refinements in a single GPU round trip. The field threshold
    /// between the passes is computed on the GPU. Returns the first pass's tip
    /// and porch levels.
    pub(crate) fn refine_field(
        &mut self,
        locations: &mut [f64],
        locked: &mut [bool],
    ) -> Result<RefinedField, String> {
        self.refine_field_inner(locations, locked)
            .map_err(|e| e.to_string())
    }
    fn refine_field_inner(
        &mut self,
        locations: &mut [f64],
        locked: &mut [bool],
    ) -> ocl::Result<RefinedField> {
        self.refine.set_arg(6, 0.0f32)?;
        let origins: Vec<_> = locations
            .iter()
            .map(|p| (*p as u64 as u32) & self.mask)
            .collect();
        let lines: Vec<_> = locations
            .iter()
            .zip(locked.iter())
            .flat_map(|(p, locked)| [p.fract() as f32, 0.0, if *locked { 1.0 } else { 0.0 }, 0.0])
            .collect();
        let mut first = vec![0f32; locations.len() * 4];
        let mut second = vec![0f32; locations.len() * 4];
        let mut threshold = [0f32; 3];
        let mut fence = UploadFence {
            queue: self._pq.queue().clone(),
            pending: true,
        };
        // All borrowed host slices outlive the final blocking read and the fence.
        unsafe {
            self.origins.write(&origins).block(false).enq()?;
            self.lines.write(&lines).block(false).enq()?;
            self.refine.enq()?;
            self.refine_threshold.enq()?;
            self.refine_at.enq()?;
            self.refined.read(&mut first).block(false).enq()?;
            self.threshold.read(&mut threshold[..]).block(false).enq()?;
        }
        self.refined_at.read(&mut second).enq()?;
        fence.pending = false;
        let apply = |locations: &mut [f64], locked: &mut [bool], result: &[f32]| {
            for ((p, valid), row) in locations
                .iter_mut()
                .zip(locked.iter_mut())
                .zip(result.as_chunks::<4>().0)
            {
                *p = p.floor() + row[0] as f64;
                *valid &= row[3] > 0.0;
            }
        };
        apply(locations, locked, &first);
        let levels = first
            .as_chunks::<4>()
            .0
            .iter()
            .zip(locked.iter())
            .map(|(v, locked)| locked.then_some((v[1], v[2])))
            .collect();
        // The CPU path also needs more than 100 locked rows for a field threshold.
        if threshold[2] > 100.0 {
            apply(locations, locked, &second);
        }
        Ok(RefinedField { levels })
    }
    /// Upload the field's lines and queue TBC rendering without waiting.
    /// With `dropouts` (first envelope sample, length, threshold fraction),
    /// dropout detection is queued too. Finish with [`Self::finish_field`].
    pub(crate) fn begin_field(
        &self,
        locations: &[f64],
        locked: &[bool],
        blank: f32,
        dropouts: Option<(u64, usize, f32)>,
    ) -> Result<FieldUpload, String> {
        let run = || -> ocl::Result<FieldUpload> {
            let (origins, lines) = self.field_lines(locations, locked);
            let upload = FieldUpload {
                fence: UploadFence {
                    queue: self._pq.queue().clone(),
                    pending: true,
                },
                _origins: origins,
                _lines: lines,
                dropouts: dropouts.map(|d| (d.1, d.2)),
            };
            if let (Some((start, count, fraction)), Some(gpu)) = (dropouts, &self.dropout_gpu) {
                if count > self.smoothing_capacity() || count == 0 {
                    return Err("Dropout envelope exceeds the smoothing buffer".into());
                }
                let chunks = count.div_ceil(DROPOUT_CHUNK);
                gpu.forward.set_arg(2, start as u32 & self.mask)?;
                gpu.forward.set_arg(3, count as u32)?;
                gpu.backward.set_arg(1, count as u32)?;
                gpu.median.set_arg(1, count as u32)?;
                gpu.chunks.set_arg(1, count as u32)?;
                gpu.chunks.set_arg(3, fraction)?;
                gpu.entries.set_arg(1, chunks as u32)?;
                gpu.events.set_arg(1, count as u32)?;
                gpu.events.set_arg(3, fraction)?;
                gpu.event_count.cmd().fill(0u32, None).enq()?;
                unsafe {
                    gpu.forward
                        .cmd()
                        .global_work_size(count.div_ceil(256))
                        .enq()?;
                    gpu.backward
                        .cmd()
                        .global_work_size(count.div_ceil(256))
                        .enq()?;
                    gpu.median.enq()?;
                    gpu.chunks.cmd().global_work_size(chunks).enq()?;
                    gpu.entries.enq()?;
                    gpu.events.cmd().global_work_size(chunks).enq()?;
                }
            }
            // The upload owns the host data until finish_field's blocking read.
            unsafe {
                self.origins.write(&upload._origins).block(false).enq()?;
                self.lines.write(&upload._lines).block(false).enq()?;
            }
            self.tbc.set_arg(5, blank)?;
            self.render_kernels()?;
            self._pq.queue().flush()?;
            Ok(upload)
        };
        run().map_err(|e| e.to_string())
    }
    /// Read the field queued by [`Self::begin_field`] and, when requested, its
    /// raw dropout intervals as [`crate::dropout::raw_intervals`] finds them.
    /// Dropouts are not repaired in the returned field.
    pub(crate) fn finish_field(
        &self,
        mut upload: FieldUpload,
    ) -> Result<(Vec<u16>, Option<Intervals>), String> {
        let run = || -> ocl::Result<(Vec<u16>, Option<Intervals>)> {
            let mut pixels = vec![0u16; WIDTH * HEIGHT * 2];
            let (Some((count, fraction)), Some(gpu)) = (upload.dropouts, &self.dropout_gpu) else {
                self.output.read(&mut pixels).enq()?;
                return Ok((pixels, None));
            };
            let mut level = [0f32; 2];
            let mut total = [0u32];
            let mut events = vec![0u32; DROPOUT_EVENTS_READ];
            // The host buffers outlive the blocking read that follows.
            unsafe {
                gpu.level.read(&mut level[..]).block(false).enq()?;
                gpu.event_count.read(&mut total[..]).block(false).enq()?;
                gpu.event_list.read(&mut events).block(false).enq()?;
            }
            self.output.read(&mut pixels).enq()?;
            let total = total[0] as usize;
            if level[1] < 32.0 {
                return Ok((pixels, Some(Vec::new())));
            }
            if total > DROPOUT_EVENTS {
                // Too many boundaries for the event list: detect on the host.
                let mut smoothed = vec![0f32; count];
                gpu.smoothed.read(&mut smoothed).len(count).enq()?;
                return Ok((
                    pixels,
                    Some(crate::dropout::raw_intervals(&smoothed, fraction)),
                ));
            }
            if total > DROPOUT_EVENTS_READ {
                events.resize(total, 0);
                gpu.event_list
                    .read(&mut events[DROPOUT_EVENTS_READ..])
                    .offset(DROPOUT_EVENTS_READ)
                    .enq()?;
            }
            events.truncate(total);
            Ok((
                pixels,
                Some(crate::dropout::event_intervals(&mut events, count)),
            ))
        };
        let result = run().map_err(|e| e.to_string())?;
        upload.fence.pending = false;
        Ok(result)
    }
    /// Repair the resident field with a dropout mask, for the GPU colour
    /// preview; the host repairs its own copy.
    /// Colour preview of the resident field after repairing it with `map`;
    /// fields from [`Self::finish_field`] are read back unrepaired.
    pub(crate) fn repaired_colour_pixels(
        &self,
        map: &crate::dropout::Map,
    ) -> Result<Vec<u8>, String> {
        let (mask, _, _) = self
            .dropout_repair
            .as_ref()
            .ok_or("Dropout repair not configured")?;
        let mut fence = UploadFence {
            queue: self._pq.queue().clone(),
            pending: true,
        };
        // The map outlives the blocking preview read, or the fence on error.
        unsafe {
            mask.write(&map.mask)
                .block(false)
                .enq()
                .map_err(|e| e.to_string())?;
        }
        self.correct_tbc().map_err(|e| e.to_string())?;
        let pixels = self.colour_pixels()?;
        fence.pending = false;
        Ok(pixels)
    }
    pub(crate) fn field(
        &mut self,
        locations: &[f64],
        locked: &[bool],
        blank: f32,
    ) -> Result<Vec<u16>, String> {
        self.field_inner(locations, locked, blank)
            .map_err(|e| e.to_string())
    }
    pub(crate) fn vbi(
        &mut self,
        locations: &[f64],
        locked: &[bool],
        blank: f32,
    ) -> Result<Vec<u16>, String> {
        let mut run = || -> ocl::Result<Vec<u16>> {
            self.upload_field_lines(locations, locked)?;
            if self.vbi.is_none() {
                let output = buffer::<u16>(&self._pq, crate::vbi::WIDTH * crate::vbi::LINES)?;
                let kernel = self
                    ._pq
                    .kernel_builder("vbi_luma")
                    .global_work_size([crate::vbi::WIDTH, crate::vbi::LINES])
                    .arg(&self.ring)
                    .arg(self.mask)
                    .arg(&self.origins)
                    .arg(&self.lines)
                    .arg(blank)
                    .arg(&output)
                    .arg(&self.sinc_lut)
                    .build()?;
                self.vbi = Some((kernel, output));
            }
            let (kernel, output) = self.vbi.as_ref().unwrap();
            kernel.set_arg(4, blank)?;
            unsafe {
                kernel.enq()?;
            }
            let mut pixels = vec![0; crate::vbi::WIDTH * crate::vbi::LINES];
            output.read(&mut pixels).enq()?;
            Ok(pixels)
        };
        run().map_err(|e| e.to_string())
    }
    fn field_inner(
        &mut self,
        locations: &[f64],
        locked: &[bool],
        blank: f32,
    ) -> ocl::Result<Vec<u16>> {
        self.render_tbc(locations, locked, blank)?;
        let mut pixels = vec![0u16; WIDTH * HEIGHT * 2];
        self.output.read(&mut pixels).enq()?;
        Ok(pixels)
    }
    fn render_tbc(&self, locations: &[f64], locked: &[bool], blank: f32) -> ocl::Result<()> {
        self.upload_field_lines(locations, locked)?;
        self.tbc.set_arg(5, blank)?;
        self.render_kernels()?;
        self.correct_tbc()?;
        Ok(())
    }
    fn render_kernels(&self) -> ocl::Result<()> {
        unsafe {
            self.burst.enq()?;
            self.chroma_state.enq()?;
            self.gain_stats.enq()?;
            self.smooth_gain.enq()?;
            self.tbc.enq()?;
        }
        Ok(())
    }
    fn correct_tbc(&self) -> ocl::Result<()> {
        if let Some((_, repaired, kernel)) = &self.dropout_repair {
            unsafe {
                kernel.enq()?;
            }
            repaired.copy(&self.output, None, None).enq()?;
        }
        Ok(())
    }
    pub(crate) fn enable_colour_preview(
        &mut self,
        full: Option<(usize, crate::preview::RasterChroma)>,
    ) -> Result<(), String> {
        self.colour_preview = Some(
            crate::export::GpuPreview::new(&self._pq, &self.output, full)
                .map_err(|e| e.to_string())?,
        );
        Ok(())
    }
    pub(crate) fn preview_colour(
        &self,
        locations: &[f64],
        locked: &[bool],
        blank: f32,
    ) -> Result<Vec<u8>, String> {
        self.render_tbc(locations, locked, blank)
            .map_err(|e| e.to_string())?;
        self.colour_pixels()
    }
    pub(crate) fn colour_pixels(&self) -> Result<Vec<u8>, String> {
        self.colour_preview
            .as_ref()
            .ok_or("Colour preview pipeline is not configured")?
            .render()
            .map_err(|e| e.to_string())
    }
    fn upload_field_lines(&self, locations: &[f64], locked: &[bool]) -> ocl::Result<()> {
        let (origins, lines) = self.field_lines(locations, locked);
        self.origins.write(&origins).enq()?;
        self.lines.write(&lines).enq()?;
        Ok(())
    }
    fn field_lines(&self, locations: &[f64], locked: &[bool]) -> (Vec<u32>, Vec<f32>) {
        let origins: Vec<_> = locations
            .iter()
            .map(|p| (*p as u64 as u32) & self.mask)
            .collect();
        let lines: Vec<_> = locations
            .iter()
            .enumerate()
            .flat_map(|(i, p)| {
                [
                    p.fract() as f32,
                    if i < HEIGHT {
                        ((locations[i + 1] - p) / WIDTH as f64) as f32
                    } else {
                        0.0
                    },
                    if locked[i] { 1.0 } else { 0.0 },
                    0.0,
                ]
            })
            .collect();
        (origins, lines)
    }
    pub(crate) fn preview(
        &mut self,
        locations: &[f64],
        locked: &[bool],
        blank: f32,
    ) -> Result<Vec<u8>, String> {
        let run = || -> ocl::Result<Vec<u8>> {
            self.upload_field_lines(locations, locked)?;
            let (kernel, output) = self
                .monitor
                .as_ref()
                .ok_or("Preview pipeline is not configured")?;
            kernel.set_arg(4, blank)?;
            unsafe {
                kernel.enq()?;
            }
            let mut pixels = vec![0u8; crate::preview::WIDTH * crate::preview::HEIGHT];
            output.read(&mut pixels).enq()?;
            Ok(pixels)
        };
        run().map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod monitor_tests {
    use super::*;
    #[test]
    #[ignore = "requires a working OpenCL GPU"]
    fn sinc_luma_matches_cpu_at_high_frequency_across_ring_wrap() {
        let fs = 315e6 / 11.0;
        let mut pipeline = Pipeline::new_monitor(fs).unwrap();
        let base = (8u64 << 32) + pipeline.mask as u64 - 1000;
        let count = (fs * 64e-6 * (HEIGHT + 1) as f64) as usize + 64;
        let signal: Vec<f32> = (0..count)
            .map(|i| 4.1e6 + 80000.0 * (std::f64::consts::TAU * 0.12 * i as f64).sin() as f32)
            .collect();
        let values: Vec<f32> = signal.iter().flat_map(|&v| [v, 4.1e6, 0.0, 0.0]).collect();
        let start = (base as u32 & pipeline.mask) as usize;
        let first = count.min(pipeline.capacity() - start);
        pipeline
            .ring
            .write(&values[..first * 4])
            .offset(start * 4)
            .enq()
            .unwrap();
        if first < count {
            pipeline
                .ring
                .write(&values[first * 4..])
                .offset(0)
                .enq()
                .unwrap();
        }
        let locations: Vec<f64> = (0..=HEIGHT)
            .map(|row| base as f64 + 16.371 + row as f64 * fs * 64e-6)
            .collect();
        let output = pipeline
            .vbi(&locations, &vec![true; HEIGHT + 1], 4.1e6)
            .unwrap();
        let mut maximum = 0;
        for row in 0..crate::vbi::LINES {
            let line = row + crate::vbi::FIRST_ROW;
            let step = (locations[line + 1] - locations[line]) / WIDTH as f64;
            for x in 0..WIDTH {
                let hz = crate::resample::sample(
                    &signal,
                    locations[line] - base as f64 + x as f64 * step,
                );
                let expected = (16384.0 + (hz - 4.1e6) * (376.32 / 7000.0)).round() as u16;
                maximum = maximum.max(output[row * WIDTH + x].abs_diff(expected));
            }
        }
        assert!(maximum <= 3, "CPU/GPU sinc code difference: {maximum}");
    }
    #[test]
    #[ignore = "requires a working OpenCL GPU"]
    fn dropout_envelope_wrap_and_pal_repair_match_cpu() {
        let fs = 315e6 / 11.0;
        let mut pipeline = Pipeline::new_monitor(fs).unwrap();
        pipeline.enable_dropouts().unwrap();
        let base = (7u64 << 32) + pipeline.mask as u64 - 500;
        let locations: Vec<f64> = (0..=HEIGHT)
            .map(|r| base as f64 + r as f64 * fs * 64e-6)
            .collect();
        let count = (locations[HEIGHT].ceil() as u64 - base) as usize;
        let mut env = vec![1.0; count];
        let hole = (100.0 * fs * 64e-6 + fs * 24e-6) as usize;
        env[hole..hole + 300].fill(0.02);
        let at = (base as u32 & pipeline.mask) as usize;
        let first = count.min(pipeline.capacity() - at);
        pipeline
            .envelope
            .write(&env[..first])
            .offset(at)
            .enq()
            .unwrap();
        pipeline
            .envelope
            .write(&env[first..])
            .offset(0)
            .enq()
            .unwrap();
        let map = pipeline.dropout_map(&locations, 0.35).unwrap();
        let expected = crate::dropout::detect(&env, base, &locations, fs, 0.35);
        assert_eq!(map.mask, expected.mask);
        let mut pixels: Vec<u16> = (0..WIDTH * HEIGHT * 2)
            .map(|i| (i % 50000) as u16)
            .collect();
        pipeline.output.write(&pixels).enq().unwrap();
        pipeline.set_dropout_mask(&map).unwrap();
        pipeline.correct_tbc().unwrap();
        crate::dropout::repair(&mut pixels, &map);
        let mut actual = vec![0; pixels.len()];
        pipeline.output.read(&mut actual).enq().unwrap();
        assert_eq!(actual, pixels, "CPU/OpenCL concealment must match exactly");
    }
    #[test]
    #[ignore = "requires a working OpenCL GPU"]
    fn gpu_dropout_intervals_match_cpu_across_ring_wrap() {
        let fs = 315e6 / 11.0;
        let mut pipeline = Pipeline::new(fs).unwrap();
        pipeline.enable_dropouts().unwrap();
        let base = (7u64 << 32) + pipeline.mask as u64 - 500;
        let locations: Vec<f64> = (0..=HEIGHT)
            .map(|r| base as f64 + r as f64 * fs * 64e-6)
            .collect();
        let count = (locations[HEIGHT].ceil() as u64 - base) as usize;
        // Noisy RF level with dropouts of many lengths, some near chunk edges.
        let mut seed = 7u32;
        let mut env: Vec<f32> = (0..count)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                0.6 + (seed >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect();
        for (n, start) in (0..count - 2000).step_by(2999).enumerate() {
            env[start..start + 5 + n % 400].fill(0.01);
        }
        env[count - 50..].fill(f32::NAN);
        let at = (base as u32 & pipeline.mask) as usize;
        let first = count.min(pipeline.capacity() - at);
        pipeline
            .envelope
            .write(&env[..first])
            .offset(at)
            .enq()
            .unwrap();
        pipeline
            .envelope
            .write(&env[first..])
            .offset(0)
            .enq()
            .unwrap();
        let locked = vec![true; HEIGHT + 1];
        let upload = pipeline
            .begin_field(&locations, &locked, 4.1e6, Some((base, count, 0.35)))
            .unwrap();
        let (_, intervals) = pipeline.finish_field(upload).unwrap();
        let expected =
            crate::dropout::raw_intervals(&crate::dropout::smooth_envelope(&env, fs), 0.35);
        assert!(expected.len() > 50, "{} intervals", expected.len());
        assert_eq!(intervals.unwrap(), expected);
    }
    #[test]
    #[ignore = "requires a working OpenCL GPU"]
    fn monitor_luma_matches_recording_without_allocating_chroma_fft_planes() {
        let fs = 315e6 / 11.0;
        let mut full = Pipeline::new(fs).unwrap();
        let mut monitor = Pipeline::new_monitor(fs).unwrap();
        assert_eq!(monitor._work[10].len(), 2);
        assert_eq!(monitor._work[11].len(), 2);
        let blocks = 32usize;
        let raw: Vec<f32> = (0..blocks * HOP + 2 * EDGE)
            .map(|i| {
                let t = i as f64 / fs;
                (std::f64::consts::TAU * 4.3e6 * t
                    + 5.0 * (std::f64::consts::TAU * 20000.0 * t).sin())
                .cos() as f32
            })
            .collect();
        // Different batch sizes must preserve the same resident luma samples.
        for pipeline in [&mut full, &mut monitor] {
            let step = pipeline.batch.min(blocks);
            for b in (0..blocks).step_by(step) {
                let count = step.min(blocks - b);
                pipeline
                    .demod(
                        &raw[b * HOP..(b + count) * HOP + 2 * EDGE],
                        (b * HOP) as u64,
                    )
                    .unwrap();
            }
        }
        let locations: Vec<f64> = (0..=HEIGHT)
            .map(|row| EDGE as f64 + fs * (0.0005 + row as f64 * 64e-6))
            .collect();
        let locked = vec![true; HEIGHT + 1];
        let tbc = full.field(&locations, &locked, 4.1e6).unwrap();
        let vbi = monitor.vbi(&locations, &locked, 4.1e6).unwrap();
        let expected_vbi = &tbc
            [crate::vbi::FIRST_ROW * WIDTH..(crate::vbi::FIRST_ROW + crate::vbi::LINES) * WIDTH];
        assert!(
            vbi.iter()
                .zip(expected_vbi)
                .all(|(a, b)| a.abs_diff(*b) <= 1),
            "VBI-only GPU path must retain full TBC sample levels and timing"
        );
        let preview = monitor.preview(&locations, &locked, 4.1e6).unwrap();
        for row in 0..crate::preview::HEIGHT {
            for x in 0..crate::preview::WIDTH {
                let sx = 185.0 + (x as f32 + 0.5) * 922.0 / crate::preview::WIDTH as f32 - 0.5;
                let col = sx as usize;
                let tx = sx - col as f32;
                let sample = |r: usize| {
                    let a = tbc[r * WIDTH + col - 1] as f32;
                    let b = tbc[r * WIDTH + col] as f32;
                    let c = tbc[r * WIDTH + col + 1] as f32;
                    let d = tbc[r * WIDTH + col + 2] as f32;
                    b + 0.5
                        * tx
                        * (c - a
                            + tx * (2.0 * a - 5.0 * b + 4.0 * c - d + tx * (3.0 * (b - c) + d - a)))
                };
                let expected = crate::preview::gray(sample(row + 22));
                assert!(preview[row * crate::preview::WIDTH + x].abs_diff(expected) <= 1);
            }
        }
    }
}
#[test]
#[ignore = "requires a working OpenCL GPU"]
fn gpu_line_clock_tracks_warp_across_ring_wrap_and_large_capture_offsets() {
    let fs = 315e6 / 11.0;
    let mut pipeline = Pipeline::new_monitor(fs).unwrap();
    let base = (3u64 << 32) + pipeline.mask as u64 - 500;
    let mut locations = vec![base as f64; HEIGHT + 1];
    let mut locked = vec![false; HEIGHT + 1];
    let mut expected = Vec::new();
    for row in 0..8 {
        let origin = base + row as u64 * 2000;
        let edge = 400.2 + 12.0 * (row as f64 * 0.71).sin();
        let mut samples = vec![0.0f32; 1600 * 4];
        for i in 0..1600 {
            let t = (i as f64 - edge) / fs;
            let pulse = if t < 0.5e-6 {
                ((t + 0.5e-6) / 1e-6).clamp(0.0, 1.0)
            } else {
                1.0 - ((t - 4.2e-6) / 1e-6).clamp(0.0, 1.0)
            };
            samples[4 * i + 1] = if row == 7 {
                4.1e6
            } else {
                (4.1e6 - 300_000.0 * pulse
                    + if t > 6e-6 {
                        30_000.0 * (row as f64).sin()
                    } else {
                        0.0
                    }) as f32
            };
        }
        let offset = (origin as u32 & pipeline.mask) as usize * 4;
        let first = samples.len().min(pipeline.ring.len() - offset);
        pipeline
            .ring
            .write(&samples[..first])
            .offset(offset)
            .enq()
            .unwrap();
        if first < samples.len() {
            pipeline
                .ring
                .write(&samples[first..])
                .offset(0)
                .enq()
                .unwrap();
        }
        locations[row] = origin as f64 + edge;
        locked[row] = true;
        expected.push(origin as f64 + edge + (2.25 / 40e6 - 3.0 / 17_734_475.0) * fs);
    }
    pipeline
        .refine_at_level(&mut locations, &mut locked, 3.95e6)
        .unwrap();
    for row in 0..7 {
        assert!(locked[row]);
        assert!(
            (locations[row] - expected[row]).abs() < 0.003,
            "row {row}: {} vs {}",
            locations[row],
            expected[row]
        );
    }
    assert!(!locked[7]);
    assert!(locked[8..].iter().all(|v| !*v));
}
