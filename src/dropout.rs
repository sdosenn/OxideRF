//! RF-envelope dropout detection and same-field PAL concealment.
use serde_json::{json, Value};
pub const WIDTH: usize = 1135;
pub const HEIGHT: usize = 313;
pub const PLANE: usize = WIDTH * HEIGHT;

pub struct Map {
    pub mask: Vec<u8>,
    spans: Vec<(usize, usize, usize)>,
}
impl Map {
    pub fn empty() -> Self {
        Self {
            mask: vec![0; PLANE],
            spans: Vec::new(),
        }
    }
    pub fn count(&self) -> usize {
        self.spans.len()
    }
    pub fn json(&self) -> Value {
        json!({"fieldLine":self.spans.iter().map(|s| s.0 + 1).collect::<Vec<_>>(),
            "startx":self.spans.iter().map(|s| s.1).collect::<Vec<_>>(),
            "endx":self.spans.iter().map(|s| s.2).collect::<Vec<_>>()})
    }
}

/// vhs-decode's default dropout threshold (`--dod_threshold_p`): 0.35 of the
/// field's RF level for ~28 MHz cxadc captures, 0.18 otherwise.
pub fn default_threshold(frequency_mhz: f64) -> f32 {
    if frequency_mhz.floor() == 28.0 {
        0.35
    } else {
        0.18
    }
}

/// Use RF amplitude, never picture brightness, to distinguish damage from whites.
pub fn detect(envelope: &[f32], base: u64, locations: &[f64], fs: f64, fraction: f32) -> Map {
    // The discriminator's analytic amplitude contains short interference dips
    // on clean FM transitions. Detect loss on a 700 kHz zero-phase envelope,
    // rather than allowing those dips to become replacement picture samples.
    detect_smoothed(
        &smooth_envelope(envelope, fs),
        base,
        locations,
        fs,
        fraction,
    )
}

/// [`detect`] on an envelope already smoothed, e.g. on the GPU.
pub fn detect_smoothed(
    envelope: &[f32],
    base: u64,
    locations: &[f64],
    fs: f64,
    fraction: f32,
) -> Map {
    map_intervals(raw_intervals(envelope, fraction), base, locations, fs)
}

/// Samples below `fraction` of the field's median smoothed RF level, until
/// the level recovers to 1.25 times that threshold. Matches the OpenCL
/// `dropout_*` kernels.
pub fn raw_intervals(envelope: &[f32], fraction: f32) -> Vec<(usize, usize)> {
    let mut levels: Vec<f32> = envelope
        .iter()
        .step_by(32)
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if levels.len() < 32 || envelope.is_empty() {
        return Vec::new();
    }
    let middle = levels.len() / 2;
    levels.select_nth_unstable_by(middle, f32::total_cmp);
    let threshold = levels[middle] * fraction;
    let mut intervals: Vec<(usize, usize)> = Vec::new();
    let mut start = None;
    for (i, &level) in envelope.iter().enumerate() {
        if !level.is_finite() || level <= threshold {
            start.get_or_insert(i);
        } else if level >= threshold * 1.25 {
            if let Some(a) = start.take() {
                intervals.push((a, i));
            }
        }
    }
    if let Some(a) = start {
        intervals.push((a, envelope.len()));
    }
    intervals
}

/// Intervals from the OpenCL boundary events: sample indices, with the top
/// bit set on the sample where the level recovered.
pub fn event_intervals(events: &mut [u32], len: usize) -> Vec<(usize, usize)> {
    events.sort_unstable_by_key(|e| e & !END);
    let mut intervals = Vec::new();
    let mut start = None;
    for &event in events.iter() {
        let at = (event & !END) as usize;
        if event & END == 0 {
            start = Some(at);
        } else if let Some(a) = start.take() {
            intervals.push((a, at));
        }
    }
    if let Some(a) = start {
        intervals.push((a, len));
    }
    intervals
}
/// Marks an OpenCL event as the end of an interval.
pub const END: u32 = 1 << 31;

/// Picture mask and spans of raw dropout intervals in a field envelope that
/// starts at absolute sample `base`.
pub fn map_intervals(intervals: Vec<(usize, usize)>, base: u64, locations: &[f64], fs: f64) -> Map {
    let mut map = Map::empty();
    let minimum = (fs * 0.35e-6).ceil() as usize;
    let merge = (fs * 1.05e-6).ceil() as usize;
    let mut joined: Vec<(usize, usize)> = Vec::new();
    // Reject brief dips before merging: otherwise a train of clean transitions
    // can join into a long false dropout across the counter of a letter.
    for (a, b) in intervals.into_iter().filter(|(a, b)| b - a >= minimum) {
        if let Some(last) = joined.last_mut().filter(|last| a - last.1 <= merge) {
            last.1 = b;
        } else {
            joined.push((a, b));
        }
    }
    for (a, b) in joined.into_iter().filter(|(a, b)| b - a >= minimum) {
        // Include the discriminator/de-emphasis transient around the RF loss.
        let begin = base as f64 + a as f64 - fs * 0.7e-6;
        let end = base as f64 + b as f64 + fs * 1.4e-6;
        for row in 0..HEIGHT {
            let left = locations[row];
            let right = locations[row + 1];
            if end <= left || begin >= right || right <= left {
                continue;
            }
            let x0 = (((begin - left) / (right - left) * WIDTH as f64).floor() as isize)
                .clamp(0, WIDTH as isize) as usize;
            let x1 = (((end - left) / (right - left) * WIDTH as f64).ceil() as isize)
                .clamp(0, WIDTH as isize) as usize;
            map.mask[row * WIDTH + x0..row * WIDTH + x1].fill(1);
        }
    }
    for row in 0..HEIGHT {
        let mut x = 0;
        while x < WIDTH {
            if map.mask[row * WIDTH + x] == 0 {
                x += 1;
                continue;
            }
            let start = x;
            while x < WIDTH && map.mask[row * WIDTH + x] != 0 {
                x += 1;
            }
            map.spans.push((row, start, x));
        }
    }
    map
}

pub(crate) fn smooth_envelope(input: &[f32], fs: f64) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let (b, a) = smoothing(fs);
    let clean: Vec<_> = input
        .iter()
        .map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 })
        .collect();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = clean.len().div_ceil(threads).max(4 * WARMUP);
    let forward = smooth_pass(&clean, b, a, false, chunk);
    smooth_pass(&forward, b, a, true, chunk)
}

/// Coefficients of the bilinear first-order 700 kHz Butterworth low-pass,
/// applied forward and backward. The CPU and the OpenCL `smooth_*` kernels
/// share this detector on field data.
pub fn smoothing(fs: f64) -> (f32, f32) {
    let k = (std::f64::consts::PI * 700_000.0 / fs).tan() as f32;
    (k / (1.0 + k), (k - 1.0) / (k + 1.0))
}

/// Samples a chunk's filter state runs through before its first output. The
/// pole is about -0.86 at 28.6 MHz, so an arbitrary start state decays below
/// f32 precision long before this and the chunks join the serial result.
pub const WARMUP: usize = 512;

/// One direction of the first-order filter, split into chunks filtered on all
/// cores. A field envelope is over half a million samples, and the serial
/// recursion otherwise dominates dropout detection.
fn smooth_pass(x: &[f32], b: f32, a: f32, reverse: bool, chunk: usize) -> Vec<f32> {
    let n = x.len();
    let mut y = vec![0.0; n];
    let step = |state: &mut (f32, f32), current: f32| {
        let value = b * (current + state.0) - a * state.1;
        *state = (current, value);
        value
    };
    std::thread::scope(|scope| {
        for (index, out) in y.chunks_mut(chunk).enumerate() {
            let begin = index * chunk;
            let end = begin + out.len();
            scope.spawn(move || {
                // At either end of the field the start state equals the serial filter's.
                if reverse {
                    let from = (end + WARMUP).min(n);
                    let mut state = (x[from - 1], x[from - 1]);
                    for &value in x[end..from].iter().rev() {
                        step(&mut state, value);
                    }
                    for (out, &value) in out.iter_mut().zip(&x[begin..end]).rev() {
                        *out = step(&mut state, value);
                    }
                } else {
                    let from = begin.saturating_sub(WARMUP);
                    let mut state = (x[from], x[from]);
                    for &value in &x[from..begin] {
                        step(&mut state, value);
                    }
                    for (out, &value) in out.iter_mut().zip(&x[begin..end]) {
                        *out = step(&mut state, value);
                    }
                }
            });
        }
    });
    y
}

pub fn repair(pixels: &mut [u16], map: &Map) -> usize {
    let original = pixels.to_vec();
    let mut repaired = 0;
    for row in 22..310 {
        for x in 185..1107 {
            let at = row * WIDTH + x;
            if map.mask[at] == 0 {
                continue;
            }
            for plane in 0..2 {
                // Four lines preserve PAL carrier phase and V-switch polarity.
                let step = if plane == 0 { 1 } else { 4 };
                for distance in (step..=16).step_by(step) {
                    let mut sum = 0u32;
                    let mut count = 0;
                    for donor in [
                        row as isize - distance as isize,
                        row as isize + distance as isize,
                    ] {
                        if (22..310).contains(&donor) && map.mask[donor as usize * WIDTH + x] == 0 {
                            sum += original[plane * PLANE + donor as usize * WIDTH + x] as u32;
                            count += 1;
                        }
                    }
                    if let Some(value) = (sum + count / 2).checked_div(count) {
                        pixels[plane * PLANE + at] = value as u16;
                        if plane == 0 {
                            repaired += 1;
                        }
                        break;
                    }
                }
            }
        }
    }
    repaired
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn brief_fm_transition_dips_do_not_merge_into_picture_damage() {
        let fs = 28e6;
        let locations: Vec<_> = (0..=HEIGHT).map(|r| r as f64 * fs * 64e-6).collect();
        let mut envelope = vec![1.0; locations[HEIGHT].ceil() as usize + 1];
        let start = (locations[120] + 700.0) as usize;
        for i in (start..start + 300).step_by(12) {
            envelope[i..i + 4].fill(0.02);
        }
        assert_eq!(detect(&envelope, 0, &locations, fs, 0.35).count(), 0);
        envelope[start + 500..start + 700].fill(0.02);
        assert_eq!(detect(&envelope, 0, &locations, fs, 0.35).count(), 1);
    }
    #[test]
    fn rf_loss_is_mapped_but_clean_bright_picture_is_not_changed() {
        let fs = 28e6;
        let locations: Vec<f64> = (0..=HEIGHT)
            .map(|r| 100.0 + r as f64 * fs * 64e-6)
            .collect();
        let mut envelope = vec![1.0; locations[HEIGHT].ceil() as usize + 1];
        let start = (locations[100] + 600.0) as usize;
        envelope[start..start + 200].fill(0.05);
        let map = detect(&envelope, 0, &locations, fs, 0.35);
        assert_eq!(map.count(), 1);
        assert_eq!(map.json()["fieldLine"][0], 101);
        let mut pixels = vec![40000; PLANE * 2];
        pixels[PLANE..].fill(32767);
        for x in 0..WIDTH {
            if map.mask[100 * WIDTH + x] != 0 {
                pixels[100 * WIDTH + x] = 65535;
            }
        }
        let before = pixels.clone();
        assert!(repair(&mut pixels, &map) > 0);
        for i in 0..PLANE {
            if map.mask[i] == 0 {
                assert_eq!(pixels[i], before[i]);
            }
        }
        assert!(pixels[100 * WIDTH + 400..100 * WIDTH + 490]
            .iter()
            .all(|p| *p == 40000));
        envelope.fill(1.0);
        assert_eq!(detect(&envelope, 0, &locations, fs, 0.35).count(), 0);
    }
    #[test]
    fn concealment_keeps_vbi_and_pal_phase_and_never_reuses_damaged_donors() {
        let mut map = Map::empty();
        map.mask[10 * WIDTH + 200] = 1;
        for row in 100..103 {
            map.mask[row * WIDTH + 200] = 1;
        }
        let mut pixels = vec![0; PLANE * 2];
        for row in 0..HEIGHT {
            for x in 0..WIDTH {
                pixels[row * WIDTH + x] = row as u16;
                pixels[PLANE + row * WIDTH + x] = 30000 + (row % 4) as u16 * 1000;
            }
        }
        let old = pixels.clone();
        repair(&mut pixels, &map);
        assert_eq!(pixels[10 * WIDTH + 200], old[10 * WIDTH + 200]);
        assert_eq!(pixels[PLANE + 101 * WIDTH + 200], 31000);
        assert_eq!(pixels[100 * WIDTH + 200], 99);
        assert_eq!(pixels[50 * WIDTH + 200], old[50 * WIDTH + 200]);
    }
    #[test]
    fn chunked_smoothing_matches_the_serial_filter_exactly() {
        let fs = 28_636_363.0;
        // A field's worth of noisy RF envelope with deep dips.
        let mut seed = 1u32;
        let input: Vec<f32> = (0..572_000)
            .map(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (seed >> 8) as f32 / (1u32 << 24) as f32;
                if i % 9000 < 40 {
                    0.02 * noise
                } else {
                    0.5 + noise
                }
            })
            .collect();
        let (b, a) = smoothing(fs);
        let serial = |x: &[f32], reverse: bool| {
            let mut y = vec![0.0; x.len()];
            let order: Vec<usize> = if reverse {
                (0..x.len()).rev().collect()
            } else {
                (0..x.len()).collect()
            };
            let mut state = (x[order[0]], x[order[0]]);
            for i in order {
                y[i] = b * (x[i] + state.0) - a * state.1;
                state = (x[i], y[i]);
            }
            y
        };
        let expected = serial(&serial(&input, false), true);
        assert_eq!(smooth_envelope(&input, fs), expected);
        // The OpenCL kernels filter 256-sample chunks.
        let gpu = smooth_pass(&smooth_pass(&input, b, a, false, 256), b, a, true, 256);
        assert_eq!(gpu, expected);
    }
}
