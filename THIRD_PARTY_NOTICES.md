# Third-party source attribution

The PAL 2D colour filter and burst reference calculations in `src/export.rs` and
`kernels/export.cl` are adapted from PALcolour in ld-decode-tools:

- Copyright 2018 William Andrew Steer
- Copyright 2018–2019 Simon Inns
- Copyright 2019 Adam Sampson
- Source: https://github.com/simoninns/ld-decode-tools/blob/main/src/ld-chroma-decoder/palcolour.cpp
- License: GNU General Public License, version 3 or later (see `LICENSE`).

The native Rust/OpenCL adaptation decodes the separate chroma TBC, preserves the
separate luma TBC, and scales/bobs each PAL field to a progressive NV12 frame.
The Y'UV to limited-range BT.601 Y'CbCr mapping follows the companion
`outputwriter.cpp` by Chad Page (2018–2021), Adam Sampson (2020–2021), and
Phillip Blucas (2021), under the same GPL-3.0-or-later license:
https://github.com/simoninns/ld-decode-tools/blob/main/src/ld-chroma-decoder/outputwriter.cpp

FFmpeg is invoked as a separate system executable for video encoding and MKV
muxing. FFmpeg and graphics drivers are not bundled with this application.

PAL VHS RF peaking parameters, output level conventions, and the peak-amplitude
ACC gain/MAD smoothing approach follow the GPL-3.0-or-later VHS-Decode reference:
https://github.com/oyvindln/vhs-decode/tree/vhs_decode/vhsdecode
Relevant sources are `format_defs/vhs.py`, `addons/biquad.py`, `chroma.py`, and
`main.py`. The native application does not execute or distribute a Python runtime.

RF-envelope dropout detection uses the VHS-Decode reference's cxadc threshold
and hysteresis conventions (0.35 and 1.25; `vhsdecode/formats.py` and
`vhsdecode/doc.py`, GPL-3.0-or-later). The native implementation in
`src/dropout.rs` uses a field median reference and independent same-field,
PAL-phase-preserving concealment; it does not execute the reference code or
claim identical detector or `ld-dropout-correct` output.

The PAL playback filter response in `src/native.rs` follows VHS-Decode's
`addons/FMdeemph.py` (`FMDeEmphasisB` / inverse high shelf), which identifies
GNU Radio / Free Software Foundation copyright 2005, 2007, 2021 and GPL-3.0-or-later.
The native frequency-domain coefficients use the PAL profile's 13.9794 dB shelf,
273755.82 Hz midpoint, Q 0.462088186, 3.4 MHz sixth-order video low-pass,
and the 65-tap Hamming 0.5 MHz sync FIR from `vhsdecode/process.py`.
These filters run in Rust/OpenCL and do not execute the reference implementation.

The 16-tap Kaiser sinc TBC reconstruction in `src/resample.rs` and
`kernels/decode.cl` follows the GPL-3.0-or-later reference's `lddecode/utils.py`
and `sinc_lut.npz` conventions (beta 5, 65536 fractional phases). The native
application generates its own coefficient table in Rust; it does not load
NumPy files or require an interpreter.
