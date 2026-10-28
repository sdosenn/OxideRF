//! Measure horizontal registration against a matched, field-aligned PAL TBC.
//! Run on short clips: both complete files are loaded into memory.
use std::{env, fs};

const WIDTH: usize = 1135;
const HEIGHT: usize = 313;
const FIELD_BYTES: usize = WIDTH * HEIGHT * 2;

fn sample(data: &[u8], field: usize, row: usize, x: usize) -> f64 {
    let offset = field * FIELD_BYTES + (row * WIDTH + x) * 2;
    u16::from_le_bytes([data[offset], data[offset + 1]]) as f64
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err(
            "Usage: compare_line_tbc CANDIDATE.tbc REFERENCE.tbc (matched short clips)".into(),
        );
    }
    let candidate = fs::read(&args[0])?;
    let reference = fs::read(&args[1])?;
    if candidate.len() % FIELD_BYTES != 0 || reference.len() % FIELD_BYTES != 0 {
        return Err("Input is not a complete 1135 x 313 uint16 PAL field stream".into());
    }
    let fields = candidate.len().min(reference.len()) / FIELD_BYTES;
    let mut residuals = Vec::new();
    let mut differences = Vec::new();
    let mut offsets = [Vec::new(), Vec::new()];
    for field in 0..fields {
        let mut shifts = Vec::new();
        for row in (30..295).step_by(2) {
            let mut scores = [0.0; 9];
            for (index, score) in scores.iter_mut().enumerate() {
                let mut ab = 0.0;
                let mut aa = 0.0;
                let mut bb = 0.0;
                for x in (235..1030).step_by(3) {
                    let shifted = x + index - 4;
                    let a = sample(&candidate, field, row, shifted + 2)
                        - sample(&candidate, field, row, shifted - 2);
                    let b = sample(&reference, field, row, x + 2)
                        - sample(&reference, field, row, x - 2);
                    ab += a * b;
                    aa += a * a;
                    bb += b * b;
                }
                *score = if aa * bb > 0.0 {
                    ab / (aa * bb).sqrt()
                } else {
                    -1.0
                };
            }
            let best = (0..9)
                .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                .unwrap();
            if !(1..8).contains(&best) || scores[best] < 0.65 {
                continue;
            }
            // Subpixel parabolic peak fit to derivative correlation. This
            // measures registration, not RF timing truth or overall quality.
            let delta = 0.5 * (scores[best - 1] - scores[best + 1])
                / (scores[best - 1] - 2.0 * scores[best] + scores[best + 1]);
            shifts.push((row, best as f64 - 4.0 + delta));
        }
        if shifts.len() < 32 {
            continue;
        }
        let center = shifts.iter().map(|(_, v)| v).sum::<f64>() / shifts.len() as f64;
        offsets[field % 2].push(center);
        residuals.extend(shifts.iter().map(|(_, v)| v - center));
        // Do not bridge rows discarded because of weak picture correlation.
        differences.extend(
            shifts
                .windows(2)
                .filter(|p| p[1].0 == p[0].0 + 2)
                .map(|p| p[1].1 - p[0].1),
        );
    }
    if residuals.is_empty() || offsets.iter().any(Vec::is_empty) {
        return Err(
            "Insufficient correlated rows; verify matching field order and clip start".into(),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "candidate": args[0], "reference": args[1], "fields": fields,
            "rows": residuals.len(), "units": "four-fSC TBC samples",
            "within_field_rms": rms(&residuals),
            "two_row_difference_rms": rms(&differences),
            "field_offsets": offsets.each_ref().map(|v| mean(v)),
            "temporal_offset_rms": offsets.each_ref().map(|v| {
                let center = mean(v);
                rms(&v.iter().map(|x| x - center).collect::<Vec<_>>())
            })
        }))?
    );
    Ok(())
}
