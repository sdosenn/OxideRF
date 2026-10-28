//! Burns teletext header row images, rendered by the VBI receiver, into
//! encoded video frames. Frames wait until the receiver has reported a header
//! from a later field, so the clock burned into each frame is the one that was
//! broadcast with it.
use crate::vbi::HeaderImage;
use std::collections::VecDeque;

/// Longest wait for the receiver: four seconds of 25 fps frames. Later frames
/// are encoded with the newest header received so far.
const MAX_DELAY_FRAMES: usize = 100;

/// Pixel layout of the frames handed to FFmpeg.
#[derive(Clone, Copy)]
pub(crate) enum Layout {
    /// Planar 10-bit 4:2:2, little endian (archive and full-raster profiles).
    Yuv422p10 { width: usize, height: usize },
    /// 8-bit 4:2:0 with interleaved chroma (viewing profile).
    Nv12 { width: usize, height: usize },
}

pub(crate) struct Burner {
    layout: Layout,
    /// Header row rectangle in frame pixels: x, y, width, height (x even).
    rect: (usize, usize, usize, usize),
    headers: VecDeque<HeaderImage>,
    pending: VecDeque<(u64, Vec<u8>)>,
    /// The scaled image of the header with this field number.
    scaled: Option<(u64, Vec<[u8; 4]>)>,
}
impl Burner {
    pub(crate) fn new(layout: Layout, rect: (usize, usize, usize, usize)) -> Self {
        Self {
            layout,
            rect,
            headers: VecDeque::new(),
            pending: VecDeque::new(),
            scaled: None,
        }
    }
    pub(crate) fn add_header(&mut self, image: HeaderImage) {
        if self
            .headers
            .back()
            .is_none_or(|last| image.field > last.field)
        {
            self.headers.push_back(image);
        }
    }
    /// Queue an encoded frame whose first field has this decoder field number.
    pub(crate) fn push_frame(&mut self, field: u64, frame: Vec<u8>) {
        self.pending.push_back((field, frame));
    }
    /// Frames ready for the encoder, in order, with their header burned in.
    pub(crate) fn ready(&mut self, finishing: bool) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Some(&(field, _)) = self.pending.front() {
            let known = self.headers.back().is_some_and(|h| h.field > field);
            if !(finishing || known || self.pending.len() > MAX_DELAY_FRAMES) {
                break;
            }
            let (field, mut frame) = self.pending.pop_front().unwrap();
            // Keep only the newest header at or before this frame, and later ones.
            while self.headers.len() > 1 && self.headers[1].field <= field {
                self.headers.pop_front();
            }
            if let Some(header) = self.headers.front().filter(|h| h.field <= field) {
                if self.scaled.as_ref().is_none_or(|(f, _)| *f != header.field) {
                    let (_, _, w, h) = self.rect;
                    let pixels = scale(&header.rgba, header.width, header.height, w, h);
                    self.scaled = Some((header.field, pixels));
                }
                let (_, pixels) = self.scaled.as_ref().unwrap();
                burn(&mut frame, self.layout, self.rect, pixels);
            }
            out.push(frame);
        }
        out
    }
}

/// Header rectangle in frame pixels for an active picture at `active_left`,
/// `active_top`, `active_width` × `active_height`; x and width are even for 4:2:2.
pub(crate) fn header_rect(
    active_left: usize,
    active_width: usize,
    active_top: usize,
    active_height: usize,
    placement: crate::preview::HeaderPlacement,
) -> (usize, usize, usize, usize) {
    let [x, y, w, h] = placement.active_rect();
    let (aw, ah) = (active_width as f32, active_height as f32);
    (
        (active_left + (x * aw).round() as usize) & !1,
        active_top + (y * ah).round() as usize,
        ((w * aw).round() as usize & !1).max(2),
        ((h * ah).round() as usize).max(2),
    )
}

/// Area-average RGBA resampling to the target size.
fn scale(rgba: &[u8], width: usize, height: usize, w: usize, h: usize) -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity(w * h);
    for ty in 0..h {
        let y0 = ty * height / h;
        let y1 = ((ty + 1) * height).div_ceil(h).clamp(y0 + 1, height);
        for tx in 0..w {
            let x0 = tx * width / w;
            let x1 = ((tx + 1) * width).div_ceil(w).clamp(x0 + 1, width);
            let mut sum = [0u32; 4];
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let at = (sy * width + sx) * 4;
                    for (s, value) in sum.iter_mut().zip(&rgba[at..at + 4]) {
                        *s += *value as u32;
                    }
                }
            }
            let count = ((y1 - y0) * (x1 - x0)) as u32;
            out.push(sum.map(|s| (s / count) as u8));
        }
    }
    out
}

/// BT.601 Y′, Cb, Cr in 0..1 / −0.5..0.5, and alpha in 0..1.
fn ycbcr(p: [u8; 4]) -> [f32; 4] {
    let [r, g, b, a] = p.map(|c| c as f32 / 255.0);
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    [y, (b - y) / 1.772, (r - y) / 1.402, a]
}

fn burn(frame: &mut [u8], layout: Layout, rect: (usize, usize, usize, usize), pixels: &[[u8; 4]]) {
    let (rx, ry, rw, rh) = rect;
    match layout {
        Layout::Yuv422p10 { width, height } => {
            let plane = width * height;
            let chroma = width / 2 * height;
            let mut blend = |index: usize, value: f32, alpha: f32| {
                let at = index * 2;
                let old = u16::from_le_bytes([frame[at], frame[at + 1]]) as f32;
                let new = (value * alpha + old * (1.0 - alpha))
                    .round()
                    .clamp(4.0, 1019.0) as u16;
                frame[at..at + 2].copy_from_slice(&new.to_le_bytes());
            };
            for ty in 0..rh.min(height.saturating_sub(ry)) {
                let y = ry + ty;
                for tx in (0..rw.min(width.saturating_sub(rx))).step_by(2) {
                    let x = rx + tx;
                    let a = ycbcr(pixels[ty * rw + tx]);
                    let b = ycbcr(pixels[ty * rw + (tx + 1).min(rw - 1)]);
                    blend(y * width + x, 64.0 + 876.0 * a[0], a[3]);
                    blend(y * width + x + 1, 64.0 + 876.0 * b[0], b[3]);
                    let alpha = (a[3] + b[3]) / 2.0;
                    let c = y * (width / 2) + x / 2;
                    blend(plane + c, 512.0 + 896.0 * (a[1] + b[1]) / 2.0, alpha);
                    blend(
                        plane + chroma + c,
                        512.0 + 896.0 * (a[2] + b[2]) / 2.0,
                        alpha,
                    );
                }
            }
        }
        Layout::Nv12 { width, height } => {
            let blend = |old: u8, value: f32, alpha: f32| {
                (value * alpha + old as f32 * (1.0 - alpha))
                    .round()
                    .clamp(1.0, 254.0) as u8
            };
            for ty in 0..rh.min(height.saturating_sub(ry)) {
                let y = ry + ty;
                for tx in 0..rw.min(width.saturating_sub(rx)) {
                    let x = rx + tx;
                    let p = ycbcr(pixels[ty * rw + tx]);
                    frame[y * width + x] = blend(frame[y * width + x], 16.0 + 219.0 * p[0], p[3]);
                    if (x | y) & 1 == 0 {
                        let mut sum = [0.0f32; 3];
                        for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                            let q =
                                ycbcr(pixels[(ty + dy).min(rh - 1) * rw + (tx + dx).min(rw - 1)]);
                            sum[0] += q[1] / 4.0;
                            sum[1] += q[2] / 4.0;
                            sum[2] += q[3] / 4.0;
                        }
                        let at = width * height + y / 2 * width + x;
                        frame[at] = blend(frame[at], 128.0 + 224.0 * sum[0], sum[2]);
                        frame[at + 1] = blend(frame[at + 1], 128.0 + 224.0 * sum[1], sum[2]);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn header(field: u64, shade: u8) -> HeaderImage {
        HeaderImage {
            field,
            width: 4,
            height: 2,
            rgba: [shade, shade, shade, 255].repeat(8),
        }
    }
    #[test]
    fn default_placement_is_teletext_row_zero() {
        let placement = crate::preview::HeaderPlacement::default();
        assert_eq!(header_rect(3, 922, 0, 576, placement), (92, 38, 744, 20));
        assert_eq!(
            header_rect(185, 922, 44, 576, placement),
            (274, 82, 744, 20)
        );
        let moved = crate::preview::HeaderPlacement {
            x: 0.5,
            y: 0.5,
            size: 2.0,
        };
        assert_eq!(header_rect(0, 720, 0, 576, moved), (360, 288, 1162, 40));
    }
    #[test]
    fn frames_wait_for_the_header_of_their_field_and_burn_it() {
        let layout = Layout::Yuv422p10 {
            width: 8,
            height: 4,
        };
        let mut burner = Burner::new(layout, (2, 1, 4, 2));
        let black = || {
            [64u16.to_le_bytes(); 32]
                .concat()
                .into_iter()
                .chain([512u16.to_le_bytes(); 32].concat())
                .collect::<Vec<u8>>()
        };
        burner.push_frame(10, black());
        burner.push_frame(12, black());
        assert!(burner.ready(false).is_empty(), "No header is known yet");
        burner.add_header(header(9, 255));
        assert!(
            burner.ready(false).is_empty(),
            "Field 10 may still get a newer header"
        );
        burner.add_header(header(11, 0));
        let ready = burner.ready(false);
        assert_eq!(ready.len(), 1, "Frame 10 uses header 9; frame 12 waits");
        let luma = |frame: &[u8], x: usize, y: usize| {
            u16::from_le_bytes([frame[(y * 8 + x) * 2], frame[(y * 8 + x) * 2 + 1]])
        };
        assert_eq!(luma(&ready[0], 2, 1), 940, "White header burned in");
        assert_eq!(luma(&ready[0], 0, 0), 64, "Outside the rectangle unchanged");
        let rest = burner.ready(true);
        assert_eq!(luma(&rest[0], 3, 2), 64, "Frame 12 uses header 11 (black)");
    }
}
