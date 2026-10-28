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
    let mut map = Map::empty();
    // The discriminator's analytic amplitude contains short interference dips
    // on clean FM transitions. Detect loss on a 700 kHz zero-phase envelope,
    // rather than allowing those dips to become replacement picture samples.
    let envelope = smooth_envelope(envelope, fs);
    let mut levels: Vec<f32> = envelope
        .iter()
        .step_by(32)
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if levels.len() < 32 || envelope.is_empty() {
        return map;
    }
    let middle = levels.len() / 2;
    levels.select_nth_unstable_by(middle, f32::total_cmp);
    let threshold = levels[middle] * fraction;
    let minimum = (fs * 0.35e-6).ceil() as usize;
    let merge = (fs * 1.05e-6).ceil() as usize;
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

fn smooth_envelope(input: &[f32], fs: f64) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    // Bilinear first-order Butterworth, applied forward and backward. Both
    // resident OpenCL and CPU decoding use this same detector on field data.
    let k = (std::f64::consts::PI * 700_000.0 / fs).tan() as f32;
    let b = k / (1.0 + k);
    let a = (k - 1.0) / (k + 1.0);
    let mut result: Vec<_> = input
        .iter()
        .map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 })
        .collect();
    let mut previous_input = result[0];
    let mut previous_output = previous_input;
    for value in &mut result {
        let current = *value;
        *value = b * (current + previous_input) - a * previous_output;
        previous_input = current;
        previous_output = *value;
    }
    previous_input = *result.last().unwrap();
    previous_output = previous_input;
    for value in result.iter_mut().rev() {
        let current = *value;
        *value = b * (current + previous_input) - a * previous_output;
        previous_input = current;
        previous_output = *value;
    }
    result
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
}
