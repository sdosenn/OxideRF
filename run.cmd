@echo off
rem Build Oxide RF when needed and start it (Windows).
rem Extra arguments are passed to the application, e.g. run.cmd --diagnose
cd /d "%~dp0"
where cargo >nul 2>nul || (echo Install Rust from https://rustup.rs first. & exit /b 1)
where ffmpeg >nul 2>nul || echo Note: FFmpeg was not found; MKV recording and export need it.
cargo run --release --locked -- %*
