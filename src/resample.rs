//! Native Kaiser-windowed sinc reconstruction for PAL TBC luma.
//! Matches the reference's 16 taps, beta 5 and 65536 fractional phases.
use std::sync::OnceLock;
pub(crate) const TAPS: usize = 16;
pub(crate) const PHASES: usize = 65536;
fn i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..50 {
        term *= x * x / (4.0 * (k * k) as f64);
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
    }
    sum
}
pub(crate) fn table() -> &'static [f32] {
    static LUT: OnceLock<Vec<f32>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = vec![0.0; (PHASES + 1) * TAPS];
        let beta = i0(5.0);
        for phase in 0..PHASES {
            let mut weights = [0.0; TAPS];
            let mut sum = 0.0;
            for (tap, w) in weights.iter_mut().enumerate() {
                let x = 7.0 - tap as f64 + phase as f64 / PHASES as f64;
                let sinc = if x.abs() < 1e-14 {
                    1.0
                } else {
                    (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                };
                let window = if x.abs() > 8.0 {
                    0.0
                } else {
                    i0(5.0 * (1.0 - (x / 8.0).powi(2)).max(0.0).sqrt()) / beta
                };
                *w = sinc * window;
                sum += *w;
            }
            for tap in 0..TAPS {
                lut[phase * TAPS + tap] = (weights[tap] / sum) as f32;
            }
        }
        // The reference LUT repeats its last phase for rounded endpoint access.
        let (rows, last) = lut.split_at_mut(PHASES * TAPS);
        last.copy_from_slice(&rows[(PHASES - 1) * TAPS..]);
        lut
    })
}
pub(crate) fn sample(v: &[f32], p: f64) -> f32 {
    let p = p.clamp(0.0, (v.len() - 1) as f64);
    let i = p.floor() as isize;
    let phase = ((p - i as f64) * PHASES as f64 + 0.5) as usize;
    let weights = &table()[phase * TAPS..(phase + 1) * TAPS];
    weights
        .iter()
        .enumerate()
        .map(|(tap, w)| v[(i + tap as isize - 7).clamp(0, v.len() as isize - 1) as usize] * w)
        .sum()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires the development reference sinc_lut.npy"]
    fn generated_coefficients_match_reference_lut() {
        let path = std::env::var("VHS_SINC_REFERENCE_LUT").unwrap();
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[..6], b"\x93NUMPY");
        let offset = 10 + u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!(bytes.len() - offset, table().len() * 4);
        let error = table()
            .iter()
            .zip(bytes[offset..].as_chunks::<4>().0)
            .map(|(a, b)| (a - f32::from_le_bytes(*b)).abs())
            .fold(0.0, f32::max);
        assert!(error < 1.2e-7, "Reference coefficient error: {error}");
    }
    #[test]
    fn reconstruction_retains_dc_sample_centers_and_band_limited_detail() {
        let v: Vec<_> = (0..512)
            .map(|i| (std::f64::consts::TAU * 0.12 * i as f64).sin() as f32)
            .collect();
        let dc = vec![4.1e6; 512];
        let mut squared = 0.0;
        for i in 24..480 {
            assert!((sample(&v, i as f64) - v[i]).abs() < 1e-5);
            let p = i as f64 + 0.371;
            assert!((sample(&dc, p) - 4.1e6).abs() < 2.0);
            squared += (sample(&v, p) - (std::f64::consts::TAU * 0.12 * p).sin() as f32).powi(2);
        }
        assert!((squared / 456.0).sqrt() < 0.001);
    }
}
