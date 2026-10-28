#!/usr/bin/env bash
# Build Oxide RF when needed and start it (Linux and macOS).
# Extra arguments are passed to the application, e.g. ./run.sh --diagnose
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v cargo >/dev/null && [[ -x "$PWD/build/toolchain/cargo/bin/cargo" ]]; then
    export RUSTUP_HOME="$PWD/build/toolchain/rustup" CARGO_HOME="$PWD/build/toolchain/cargo"
    export PATH="$CARGO_HOME/bin:$PATH"
fi
command -v cargo >/dev/null || { echo 'Install Rust from https://rustup.rs first.' >&2; exit 1; }
command -v ffmpeg >/dev/null || echo 'Note: FFmpeg was not found; MKV recording and export need it.' >&2
exec cargo run --release --locked -- "$@"
