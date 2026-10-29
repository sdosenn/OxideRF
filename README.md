# Oxide RF

Cross platform GUI PAL VHS RF decoder. Built using AI tools.<br>
Capture RF, decode and record VHS tapes in real time.

<!-- <p align="left">
  <a href="https://github.com/sdosenn/OxideRF/releases/latest">
    <img src="https://img.shields.io/badge/Download_latest_release-0.3.2--beta-2ea44f?style=for-the-badge&logo=github" alt="Download Oxide RF 0.3.2-beta">
  </a>
</p>-->

<!-- Screenshot of the main window -->

## Realtime RF capture and decoding chain

Decode the RF while the tape plays and see the picture at the same moment.<br>
No need to store hundreds of gigabytes of RF and decode it later. Do it all in one pass.<br>
Record a ready-to-watch MKV directly, or vhs-decode compatible TBC files when you
want to reprocess later.<br>

## ☕ Support the project

Oxide RF is built to make VHS RF capture as simple as playing a tape. If it
helps you rescue a few more recordings, consider supporting its continued
development.

Your support helps me spend more time on better decoding, faster and simpler
capture workflows, broader hardware compatibility, and reliable cross-platform
releases.

<p align="center">
  <a href="https://ko-fi.com/sinisinavideoteka">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support Oxide RF on Ko-fi" height="36">
  </a>
</p>

Every recovered tape is a small piece of history that might otherwise
disappear. Thank you for helping keep the work moving.

## Highlights

- Decode PAL VHS RF on the GPU with OpenCL.
- Read RF files, capture live from a cxadc card, or receive RF from another PC.
- Watch the decoded picture live, in luma only or colour.
- Record lossless FFV1 or H.264 MKV directly, without intermediate files.
- Write vhs-decode compatible TBC files (luma, chroma and JSON) for later processing.
- Stream VBI to [TeletextRecoveReese](https://github.com/sdosenn/TeletextRecoveReese).
- Runs on Linux, Windows, and macOS.

## Sources

- **RF file** — open or drop an RF capture (see [File formats](#file-formats)).
- **Live cxadc** — capture from a CX2388x card with the
  [cxadc](https://github.com/happycube/cxadc-linux3) driver on Linux or
  [cxadc-win](https://github.com/JuniorIsAJitterbug/cxadc-win) on Windows.
- **Network RF** — receive RF from an Oxide RF server on another PC, e.g.
  `tcp://192.168.1.50:5555`.
- **RF server** — send a local cxadc card to a decoding PC.

## Recording

- **Preview only** — watch the picture; nothing is written to disk.
- **TBC files** — write `name.tbc`, `name_chroma.tbc` and `name.tbc.json`.
  **Export MKV after decoding TBC** converts them to MKV when recording stops.
- **Direct MKV** — encode `name.mkv` while decoding, without TBC files.

## Preview

- **4:3 picture** — the active picture, fields woven into 720 × 576.
- **Full raster** — the whole TBC field, 1135 samples × 313 lines, including
  sync, colour burst and VBI lines with teletext.
- **Luma only** — the lightest GPU path, grayscale.
- **Chroma + luma** — PAL colour, decoded the same way as MKV export.

## VCR audio

- This captures the VCR's baseband audio output; it does not decode VHS Hi-Fi
audio from the RF.

## Teletext / VBI

- **Stream VBI over TCP** sends the raw VBI lines of every decoded field to a
  teletext decoder, normally on port 5556.
- In TeletextRecoveReese, choose **File → Open VBI capture → From TCP stream…**
  and the **VHS-decode VBI-only TBC** preset.
- TeletextRecoveReese sends back the decoded header row. With **Burn teletext
  header into the MKV**, Direct MKV burns into each frame the header that was
  broadcast with it, so the burned clock matches the recording.

## Settings

- **RF buffer → RAM limit** — how much memory live RF may queue while the
  decoder is behind, 4 GiB by default (about 150 s at 28.636 MHz). A larger
  buffer absorbs longer stalls; it cannot fix headroom below 1×.

Settings are stored in `~/.config/oxide-rf` on Linux, `%APPDATA%\oxide-rf` on
Windows and `~/Library/Application Support/oxide-rf` on macOS.

## Platform support

| Feature | Linux | Windows | macOS |
| --- | --- | --- | --- |
| RF file decoding, TBC, MKV, full raster | yes | yes | yes |
| Network RF and server audio | yes | yes | yes |
| Live cxadc capture and RF server | yes (cxadc) | yes (cxadc-win) | no driver |
| Sending VCR audio from the RF server | yes (PulseAudio) | yes (WASAPI) | no driver |
| Teletext/VBI over TCP and header burn-in | yes | yes | yes |

On Windows, set input, gain and 10-bit mode with cxadc-win's PowerShell module
(`Set-CxadcWinConfig`) and enter the sample rate manually; the driver does not
know the crystal rate.

## File formats

### RF input

| Format | Extension / selection |
| --- | --- |
| Unsigned 8-bit cxadc | `.r8`, `.u8` or `u8` |
| Unsigned 16-bit little endian | `.r16`, `.u16` or `u16` |
| Signed PCM | `.s8`, `.s16` |
| Float32 little endian | `.rf` or `f32` |
| Mono RF FLAC | `.flac`, `.ldf`, `.vhs` |
| Mono PCM/float WAV | `.wav` |
| Generic `.raw` | Select the sample format manually |

FLAC and WAV use the sample rate selected in the GUI, not the container's.
Packed `.lds`/`.r30` input is not supported.

### TBC output

- `name.tbc` — luma, 1135 × 313 UInt16 little-endian samples per field.
- `name_chroma.tbc` — chroma with the same geometry.
- `name.tbc.json` — vhs-decode style metadata: levels, field order, sync and dropouts.

### VBI over TCP

- No header or handshake; each field is 16 × 1135 UInt16 little-endian samples
  (36320 bytes), TBC rows 6–21 (PAL lines 7–22).

## Download and run

### Linux AppImage

```bash
chmod +x Oxide-RF-0.3.2-beta-x86_64.AppImage
./Oxide-RF-0.3.2-beta-x86_64.AppImage
```

Needs x86_64 Linux with glibc 2.39 or newer (Ubuntu 24.04 or later) and an
OpenCL driver for the GPU, e.g. `mesa-opencl-icd` or `intel-opencl-icd`. Without
FUSE, add `--appimage-extract-and-run`. Without a GPU, turn off **OpenCL GPU
acceleration** to decode on the CPU.

## Building from source

Install [Rust](https://rustup.rs), an OpenCL driver for the GPU and
[FFmpeg](https://ffmpeg.org) in `PATH`, then:

| System | Command |
| --- | --- |
| Linux, macOS | `./run.sh` |
| Windows | `run.cmd` |
| any | `cargo run --release` |

- **Linux:** `sudo apt install build-essential pkg-config libasound2-dev ocl-icd-opencl-dev`
  (Debian/Ubuntu) or `sudo pacman -S base-devel alsa-lib ocl-icd opencl-headers` (Arch).
- **Windows:** Visual Studio Build Tools with the C++ workload, and `OpenCL.lib`
  from [vcpkg](https://vcpkg.io) (`vcpkg install opencl:x64-windows`) or a folder
  set in `OPENCL_LIB_DIR`.
- **macOS:** Xcode Command Line Tools (`xcode-select --install`).

Run the tests:

```bash
cargo test --release --lib --bin oxide-rf
cargo clippy --all-targets --all-features -- -D warnings
```

Build the AppImage with `scripts/build-appimage.sh` after placing
`appimagetool-x86_64.AppImage` and `runtime-x86_64` in `build/tools/`. The About
picture is embedded from `assets/about.png` or `assets/about.jpg` (401 × 500).


## Status

- Current development version: **0.3.2-beta**
- Experimental; actively developed and used for real captures.
- Equal quality to vhs-decode has not been established.
- Licensed under GPL-3.0-or-later.

## Thanks and inspiration

Oxide RF would not have been possible without the ideas, research, and
practical solutions of these projects:

- **[vhs-decode](https://github.com/oyvindln/vhs-decode)** — the reference VHS RF
  decoder, its TBC format, filter profiles and dropout detection conventions.
- **[ld-decode](https://github.com/happycube/ld-decode)** — the RF decoding
  foundations and the ld-chroma-decoder PAL colour filter adapted here.
- **[cxadc-linux3](https://github.com/happycube/cxadc-linux3)** and
  **[cxadc-win](https://github.com/JuniorIsAJitterbug/cxadc-win)** — the drivers
  that turn a cheap capture card into an RF digitizer.
- **[tbc-video-export](https://github.com/JuniorIsAJitterbug/tbc-video-export)** —
  the archival MKV workflow the export profile follows.

Thanks to their respective authors and contributors for solving hard problems
and sharing their work with the tape preservation community. Detailed
attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Built with [egui/eframe](https://github.com/emilk/egui),
[ocl](https://github.com/cogciprocate/ocl), [cpal](https://github.com/RustAudio/cpal),
[rfd](https://github.com/PolyMeilex/rfd), [claxon](https://github.com/ruuda/claxon),
[hound](https://github.com/ruuda/hound) and [image](https://github.com/image-rs/image),
with [FFmpeg](https://ffmpeg.org) for MKV encoding.
