use std::{fs::File, io::Read, path::Path};
pub const WIDTH: usize = 720;
pub const HEIGHT: usize = 288;
/// Full-raster preview field: every TBC sample (1135, padded to an even width)
/// and every line (313, padded to an even height for NV12).
pub const FULL_WIDTH: usize = 1136;
pub const FULL_HEIGHT: usize = 314;
/// Display aspect of a full-raster frame: 922 active samples × 576 lines are
/// 4:3, so each sample is 384/461 of a line pitch wide.
pub const FULL_ASPECT: f32 = 1135.0 * 384.0 / 461.0 / 626.0;
/// How chroma is shown in the full raster.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RasterChroma {
    /// PAL colour in the active picture only; neutral elsewhere.
    Active,
    /// PAL colour decoded across the whole raster, including burst and blanking.
    Whole,
    /// Luma plus the raw chroma signal in grey: burst and subcarrier as recorded.
    Composite,
}
impl RasterChroma {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "active" => Some(Self::Active),
            "whole" => Some(Self::Whole),
            "composite" => Some(Self::Composite),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Whole => "whole",
            Self::Composite => "composite",
        }
    }
    /// Mode number used by the OpenCL kernels.
    pub fn code(self) -> i32 {
        self as i32
    }
}
/// Display aspect for a woven preview frame of the given size.
pub fn display_aspect(width: usize) -> f32 {
    if width == WIDTH {
        4.0 / 3.0
    } else {
        FULL_ASPECT
    }
}
pub const BLACK: f32 = 14_745.6;
pub const WHITE: f32 = 59_417.6;

pub fn gray(code: f32) -> u8 {
    ((code - BLACK) * 255.0 / (WHITE - BLACK)).clamp(0.0, 255.0) as u8
}

pub struct PreviewFrame {
    pub width: usize,
    pub height: usize,
    pub field: u64,
    pub pixels: Vec<u8>,
    pub first_field: Option<bool>,
    pub colour: bool,
}

impl PreviewFrame {
    pub fn read(path: &Path) -> Result<Self, String> {
        let mut bytes = Vec::new();
        File::open(path)
            .map_err(|e| e.to_string())?
            .take(1920 * 1080 * 3 + 29)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        Self::parse(&bytes)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 24
            || ![b"VHSPRV01", b"VHSPRV02", b"VHSPRV03", b"VHSPRV04"]
                .contains(&bytes[..8].try_into().unwrap())
        {
            return Err("Invalid preview format".into());
        }
        let width = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let height = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let field = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let nv12 = &bytes[..8] == b"VHSPRV03";
        let colour = nv12 || &bytes[..8] == b"VHSPRV04";
        let (header, first_field) = if &bytes[..8] == b"VHSPRV02" || nv12 {
            if bytes.len() < 28 {
                return Err("Incomplete field preview header".into());
            }
            let flags = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
            if flags > 1 {
                return Err("Invalid preview field polarity".into());
            }
            (28, Some(flags == 1))
        } else {
            (24, None)
        };
        if width == 0
            || height == 0
            || width > 1920
            || height > 1080
            || (nv12 && (!width.is_multiple_of(2) || !height.is_multiple_of(2)))
            || bytes.len()
                != header
                    + if nv12 {
                        width * height * 3 / 2
                    } else {
                        width * height * if colour { 3 } else { 1 }
                    }
        {
            return Err("Incomplete preview image".into());
        }
        Ok(Self {
            width,
            height,
            field,
            pixels: if nv12 {
                nv12_to_rgb(width, height, &bytes[header..])
            } else {
                bytes[header..].to_vec()
            },
            first_field,
            colour,
        })
    }
}
#[derive(Default)]
pub struct FrameAssembler {
    previous: Option<PreviewFrame>,
}
impl FrameAssembler {
    pub fn clear(&mut self) {
        self.previous = None;
    }
    pub fn push(&mut self, frame: PreviewFrame) -> Option<PreviewFrame> {
        let Some(first) = frame.first_field else {
            self.clear();
            return Some(frame);
        };
        let complete = self.previous.take().filter(|p| {
            p.field.checked_add(1) == Some(frame.field)
                && p.width == frame.width
                && p.height == frame.height
                && p.first_field == Some(!first)
                && p.colour == frame.colour
        });
        let result = complete.map(|previous| {
            let (top, bottom) = if first {
                (&frame, &previous)
            } else {
                (&previous, &frame)
            };
            let mut pixels = vec![0; frame.pixels.len() * 2];
            for row in 0..frame.height {
                let stride = frame.width * if frame.colour { 3 } else { 1 };
                let start = row * stride;
                pixels[2 * start..2 * start + stride]
                    .copy_from_slice(&top.pixels[start..start + stride]);
                pixels[2 * start + stride..2 * start + 2 * stride]
                    .copy_from_slice(&bottom.pixels[start..start + stride]);
            }
            PreviewFrame {
                width: frame.width,
                height: frame.height * 2,
                field: frame.field,
                pixels,
                first_field: None,
                colour: frame.colour,
            }
        });
        self.previous = Some(frame);
        result
    }
}
pub fn field_packet(field: u64, first: bool, pixels: &[u8]) -> Vec<u8> {
    sized_field_packet(WIDTH, HEIGHT, field, first, pixels)
}
pub fn sized_field_packet(
    width: usize,
    height: usize,
    field: u64,
    first: bool,
    pixels: &[u8],
) -> Vec<u8> {
    let mut result = packet(width as u32, height as u32, field, &[]);
    result[..8].copy_from_slice(b"VHSPRV02");
    result.extend(u32::from(first).to_le_bytes());
    result.extend(pixels);
    result
}

pub fn colour_field_packet(
    width: usize,
    height: usize,
    field: u64,
    first: bool,
    nv12: &[u8],
) -> Vec<u8> {
    let mut result = sized_field_packet(width, height, field, first, nv12);
    result[..8].copy_from_slice(b"VHSPRV03");
    result
}

/// Teletext header row image for the GUI overlay: `VHSHDR01`, width, height
/// (u32), decoder field (u64), then straight-alpha RGBA.
pub fn header_packet(width: usize, height: usize, field: u64, rgba: &[u8]) -> Vec<u8> {
    let mut result = packet(width as u32, height as u32, field, rgba);
    result[..8].copy_from_slice(b"VHSHDR01");
    result
}
/// Where the teletext header sits, relative to the active picture: the top-left
/// corner as fractions of its width and height, and a size factor. The default
/// is where a TV shows row 0: 40 characters across about 42 µs of the 52 µs
/// active line, centred, and 20 of 576 lines high, 38 lines from the top.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct HeaderPlacement {
    pub x: f32,
    pub y: f32,
    pub size: f32,
}
impl Default for HeaderPlacement {
    fn default() -> Self {
        Self {
            x: (1.0 - 42.0 / 52.0) / 2.0,
            y: 38.0 / 576.0,
            size: 1.0,
        }
    }
}
impl HeaderPlacement {
    pub const SIZE_RANGE: std::ops::RangeInclusive<f32> = 0.25..=3.0;
    pub fn is_valid(self) -> bool {
        (0.0..=1.0).contains(&self.x)
            && (0.0..=1.0).contains(&self.y)
            && Self::SIZE_RANGE.contains(&self.size)
    }
    /// The same placement moved and, if needed, shrunk so the header lies
    /// entirely inside the active picture.
    pub fn fitted(self) -> Self {
        let [_, _, w, h] = Self { size: 1.0, ..self }.active_rect();
        let size = self
            .size
            .min(1.0 / w)
            .min(1.0 / h)
            .clamp(*Self::SIZE_RANGE.start(), *Self::SIZE_RANGE.end());
        Self {
            x: self.x.clamp(0.0, (1.0 - w * size).max(0.0)),
            y: self.y.clamp(0.0, (1.0 - h * size).max(0.0)),
            size,
        }
    }
    /// x, y, width and height as fractions of the active picture.
    pub fn active_rect(self) -> [f32; 4] {
        [
            self.x,
            self.y,
            42.0 / 52.0 * self.size,
            20.0 / 576.0 * self.size,
        ]
    }
}
/// A parsed teletext header overlay image.
pub struct HeaderOverlay {
    pub width: usize,
    pub height: usize,
    pub field: u64,
    pub rgba: Vec<u8>,
}
impl HeaderOverlay {
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 24 || &bytes[..8] != b"VHSHDR01" {
            return None;
        }
        let width = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
        let height = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
        let field = u64::from_le_bytes(bytes[16..24].try_into().ok()?);
        (width > 0
            && height > 0
            && width <= 2048
            && height <= 256
            && bytes.len() == 24 + width * height * 4)
            .then(|| Self {
                width,
                height,
                field,
                rgba: bytes[24..].to_vec(),
            })
    }
    /// The header rectangle in a woven preview frame, as fractions of the frame;
    /// the same placement is used when burning it into the MKV.
    pub fn placement(frame_width: usize, placement: HeaderPlacement) -> [f32; 4] {
        let [x, y, w, h] = placement.active_rect();
        let [left, top, width, height] = Self::active_area(frame_width);
        [left + x * width, top + y * height, w * width, h * height]
    }
    /// The active picture as fractions of a woven preview frame: x, y, width
    /// and height.
    pub fn active_area(frame_width: usize) -> [f32; 4] {
        if frame_width == WIDTH {
            [0.0, 0.0, 1.0, 1.0]
        } else {
            // Full raster: the active picture is samples 185..1107, lines 44..620.
            [185.0 / 1135.0, 44.0 / 626.0, 922.0 / 1135.0, 576.0 / 626.0]
        }
    }
}
pub fn frame_packet(frame: &PreviewFrame) -> Vec<u8> {
    let mut result = packet(
        frame.width as u32,
        frame.height as u32,
        frame.field,
        &frame.pixels,
    );
    if frame.colour {
        result[..8].copy_from_slice(b"VHSPRV04");
    }
    result
}

fn nv12_to_rgb(width: usize, height: usize, data: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(width * height * 3);
    for row in 0..height {
        for x in 0..width {
            let y = (data[row * width + x] as f32 - 16.0) * (255.0 / 219.0);
            let at = width * height + (row / 2) * width + (x & !1);
            let u = (data[at] as f32 - 128.0) * (255.0 / 224.0);
            let v = (data[at + 1] as f32 - 128.0) * (255.0 / 224.0);
            rgb.extend(
                [
                    y + 1.402 * v,
                    y - 0.344_136 * u - 0.714_136 * v,
                    y + 1.772 * u,
                ]
                .map(|c| c.round().clamp(0.0, 255.0) as u8),
            );
        }
    }
    rgb
}

pub fn packet(width: u32, height: u32, field: u64, pixels: &[u8]) -> Vec<u8> {
    let mut packet = Vec::with_capacity(24 + pixels.len());
    packet.extend(b"VHSPRV01");
    packet.extend(width.to_le_bytes());
    packet.extend(height.to_le_bytes());
    packet.extend(field.to_le_bytes());
    packet.extend(pixels);
    packet
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_levels_keep_superblack_and_superwhite_headroom() {
        assert_eq!(gray(BLACK), 0);
        assert_eq!(gray(WHITE), 255);
        assert!(gray(16384.0) > 0);
        assert!(gray(54016.0) < 255);
    }
    #[test]
    fn full_frame_weaves_field_rows_without_vertical_bounce() {
        let field = |seq, first, pixels: Vec<u8>| PreviewFrame {
            width: 2,
            height: 2,
            field: seq,
            pixels,
            first_field: Some(first),
            colour: false,
        };
        for first in [true, false] {
            let mut assembler = FrameAssembler::default();
            let top = vec![10, 11, 30, 31];
            let bottom = vec![20, 21, 40, 41];
            assert!(assembler
                .push(field(
                    1,
                    first,
                    if first { top.clone() } else { bottom.clone() }
                ))
                .is_none());
            let frame = assembler
                .push(field(2, !first, if first { bottom } else { top }))
                .unwrap();
            assert_eq!((frame.width, frame.height), (2, 4));
            assert_eq!(frame.pixels, vec![10, 11, 20, 21, 30, 31, 40, 41]);
            assert!(
                assembler.push(field(4, first, vec![0; 4])).is_none(),
                "Missing fields must not be woven into stale pictures"
            );
            assert!(
                assembler.push(field(5, first, vec![0; 4])).is_none(),
                "Repeated polarity must not create a frame"
            );
        }
    }
    #[test]
    fn validates_packet_and_preserves_gray_levels() {
        let mut data = b"VHSPRV01".to_vec();
        data.extend(2u32.to_le_bytes());
        data.extend(1u32.to_le_bytes());
        data.extend(12u64.to_le_bytes());
        data.extend([0, 255]);
        let p = PreviewFrame::parse(&data).unwrap();
        assert_eq!((p.width, p.height, p.field), (2, 1, 12));
        assert_eq!(p.pixels, [0, 255]);
        data.pop();
        assert!(PreviewFrame::parse(&data).is_err());
        assert!(PreviewFrame::parse(b"invalid").is_err());
    }
    #[test]
    fn colour_transport_decodes_bt601_and_validates_payload_geometry() {
        let count = WIDTH * HEIGHT;
        let mut nv12 = vec![81; count];
        nv12.extend([90, 240].repeat(count / 4));
        let mut data = colour_field_packet(WIDTH, HEIGHT, 7, true, &nv12);
        let frame = PreviewFrame::parse(&data).unwrap();
        assert!(frame.colour);
        assert_eq!(frame.first_field, Some(true));
        assert_eq!(frame.pixels.len(), count * 3);
        for rgb in frame.pixels.as_chunks::<3>().0 {
            assert!(rgb[0] >= 253 && rgb[1] <= 2 && rgb[2] <= 2);
        }
        data.pop();
        assert!(PreviewFrame::parse(&data).is_err());
        let mut odd = colour_field_packet(WIDTH, HEIGHT, 7, true, &nv12);
        odd[8..12].copy_from_slice(&719u32.to_le_bytes());
        assert!(PreviewFrame::parse(&odd).is_err());
        let mut invalid = colour_field_packet(WIDTH, HEIGHT, 7, true, &nv12);
        invalid[24..28].copy_from_slice(&2u32.to_le_bytes());
        assert!(PreviewFrame::parse(&invalid).is_err());
    }
    #[test]
    fn rgb_field_assembly_preserves_polarity_and_resets_on_format_changes() {
        let field = |seq, first, colour, pixels| PreviewFrame {
            width: 2,
            height: 2,
            field: seq,
            pixels,
            first_field: Some(first),
            colour,
        };
        let mut assembler = FrameAssembler::default();
        assert!(assembler
            .push(field(1, true, true, vec![255; 12]))
            .is_none());
        let frame = assembler.push(field(2, false, true, vec![0; 12])).unwrap();
        assert_eq!(
            frame.pixels,
            [vec![255; 6], vec![0; 6], vec![255; 6], vec![0; 6]].concat()
        );
        let parsed = PreviewFrame::parse(&frame_packet(&frame)).unwrap();
        assert!(parsed.colour);
        assert_eq!(parsed.pixels, frame.pixels);
        assert_eq!((parsed.width, parsed.height), (2, 4));
        assert!(assembler.push(field(3, true, false, vec![0; 4])).is_none());
    }
}
