use eframe::egui;
use std::{
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};
use vhs_opencl::preview;

mod settings;
mod theme;

#[derive(Clone, Copy, PartialEq)]
enum Job {
    Decode,
    Export,
    Server,
}

struct App {
    input: String,
    live: bool,
    network: bool,
    network_input: String,
    server: bool,
    server_bind: String,
    live_input: String,
    cxadc_devices: Vec<vhs_opencl::cxadc::DeviceInfo>,
    output: String,
    frequency: f64,
    frames: u32,
    start: u32,
    sample_format: String,
    gpu: bool,
    devices: Vec<String>,
    device: usize,
    child: Option<Child>,
    rx: Option<mpsc::Receiver<String>>,
    log: Vec<String>,
    progress: Option<String>,
    status: String,
    started: Option<Instant>,
    stopping: bool,
    job: Job,
    export_after_decode: bool,
    direct_mkv: bool,
    export_encoder: String,
    export_profile: String,
    vbi_enabled: bool,
    vbi_bind: String,
    vbi_wait: bool,
    dropout_correct: bool,
    /// `None` follows vhs-decode's default for the sample rate.
    dropout_threshold: Option<f32>,
    server_audio: bool,
    audio_source: String,
    audio_devices: Vec<vhs_opencl::audio::Device>,
    audio_bind: String,
    audio_monitor: bool,
    audio_offset_ms: f64,
    pending_export: Option<String>,
    preview_path: Option<PathBuf>,
    preview_texture: Option<egui::TextureHandle>,
    preview_field: u64,
    preview_checked: Instant,
    preview_error: Option<String>,
    preview_only: bool,
    preview_colour: bool,
    /// Preview and record the full TBC raster instead of the active picture.
    full_raster: bool,
    /// Burn the teletext header returned by the VBI receiver into the MKV.
    burn_teletext: bool,
    /// Where the teletext header is shown and burned, relative to the picture.
    header_placement: preview::HeaderPlacement,
    /// Upper limit of the RAM queue that holds RF while decoding lags behind.
    live_buffer_mib: u32,
    show_settings: bool,
    show_about: bool,
    about_texture: Option<egui::TextureHandle>,
    /// Workspace height around the preview picture in the last frame.
    workspace_chrome: f32,
    /// Width the start/stop button and preview switches need on one row.
    toolbar_width: f32,
    /// Latest teletext header image from the VBI receiver, for the overlay.
    header_texture: Option<egui::TextureHandle>,
    preview_socket: Option<vhs_opencl::channel::PreviewReceiver>,
    /// Jobs stop gracefully when "stop" is written here (all platforms).
    child_stdin: Option<ChildStdin>,
    frame_assembler: preview::FrameAssembler,
    automatic_black: bool,
    /// Latest decoder rate and headroom from `PREVIEW_RATE`.
    metrics: Option<(f64, f64)>,
    log_open: bool,
    /// Normal (not maximized) window size in points, maximized state and the
    /// settings column width, restored at the next start.
    window_size: Option<[f32; 2]>,
    window_maximized: bool,
    /// Last settings written to disk, and when the current ones first differed.
    saved_settings: serde_json::Value,
    settings_changed: Option<Instant>,
    rf_queue_s: Option<f64>,
    #[cfg(feature = "ui-screenshot")]
    preview_uploads: u64,
}
impl Default for App {
    fn default() -> Self {
        Self {
            input: String::new(),
            live: false,
            network: false,
            network_input: "tcp://192.168.1.50:5555".into(),
            server: false,
            server_bind: "0.0.0.0:5555".into(),
            live_input: vhs_opencl::cxadc::DEFAULT_DEVICE.into(),
            cxadc_devices: vhs_opencl::cxadc::devices(),
            output: String::new(),
            frequency: 315.0 / 11.0,
            frames: 0,
            start: 0,
            sample_format: "auto".into(),
            gpu: true,
            devices: vhs_opencl::devices(),
            device: 0,
            child: None,
            rx: None,
            log: Vec::new(),
            progress: None,
            status: "Ready".into(),
            started: None,
            stopping: false,
            job: Job::Decode,
            export_after_decode: false,
            direct_mkv: false,
            export_encoder: "auto".into(),
            export_profile: "archive".into(),
            vbi_enabled: false,
            vbi_bind: "0.0.0.0:5556".into(),
            vbi_wait: true,
            dropout_correct: false,
            dropout_threshold: None,
            server_audio: false,
            audio_source: "default".into(),
            audio_devices: vhs_opencl::audio::devices(),
            audio_bind: "0.0.0.0:5557".into(),
            audio_monitor: true,
            audio_offset_ms: 0.0,
            pending_export: None,
            preview_path: None,
            preview_texture: None,
            preview_field: 0,
            preview_checked: Instant::now(),
            preview_error: None,
            preview_only: false,
            preview_colour: false,
            full_raster: false,
            burn_teletext: false,
            header_placement: preview::HeaderPlacement::default(),
            live_buffer_mib: vhs_opencl::cxadc::DEFAULT_BUFFER_MIB as u32,
            show_settings: false,
            show_about: false,
            about_texture: None,
            workspace_chrome: 160.0,
            toolbar_width: 800.0,
            header_texture: None,
            preview_socket: None,
            child_stdin: None,
            frame_assembler: Default::default(),
            automatic_black: false,
            metrics: None,
            log_open: false,
            window_size: None,
            window_maximized: false,
            saved_settings: serde_json::Value::Null,
            settings_changed: None,
            rf_queue_s: None,
            #[cfg(feature = "ui-screenshot")]
            preview_uploads: 0,
        }
    }
}
impl App {
    fn choose_input(&mut self, path: PathBuf) {
        self.live = false;
        self.network = false;
        self.server = false;
        self.output = path.with_extension("").to_string_lossy().to_string() + "-decoded";
        self.input = path.to_string_lossy().into();
    }
    fn refresh_cxadc(&mut self, select_first: bool) {
        self.cxadc_devices = vhs_opencl::cxadc::devices();
        if select_first {
            if !self
                .cxadc_devices
                .iter()
                .any(|d| d.path == std::path::Path::new(&self.live_input))
            {
                if let Some(device) = self.cxadc_devices.first() {
                    self.live_input = device.path.to_string_lossy().into_owned();
                }
            }
            self.apply_cxadc_settings();
        }
    }
    fn apply_cxadc_settings(&mut self) {
        let device = vhs_opencl::cxadc::inspect(std::path::Path::new(&self.live_input));
        if let Some(format) = device.sample_format() {
            self.sample_format = format.into();
        }
        if let Some(frequency) = device.frequency_mhz {
            self.frequency = frequency;
        }
    }
    fn launch(&mut self) -> Result<(), String> {
        if self.server {
            return self.launch_server();
        }
        let input = if self.network {
            &self.network_input
        } else if self.live {
            &self.live_input
        } else {
            &self.input
        };
        if self.network {
            if !input.starts_with("tcp://") {
                return Err("Enter tcp://HOST:PORT for the native RF sender".into());
            }
        } else if self.live {
            if !vhs_opencl::cxadc::is_stream(std::path::Path::new(input)) {
                return Err(format!(
                    "Select a cxadc device, for example {}",
                    vhs_opencl::cxadc::DEFAULT_DEVICE
                ));
            }
        } else if !PathBuf::from(input).is_file() {
            return Err("Select an RF capture".into());
        }
        if !self.preview_only && self.output.trim().is_empty() {
            return Err("Select an output base name".into());
        }
        if !self.preview_only
            && (self.export_after_decode || self.direct_mkv)
            && PathBuf::from(format!("{}.mkv", self.output)).exists()
        {
            return Err("The MKV output already exists. Choose another output base name.".into());
        }
        if !self.live
            && !self.network
            && self.input.to_lowercase().ends_with(".raw")
            && self.sample_format == "auto"
        {
            return Err(
                "For .raw, select the sample format (cxadc: u8 or u16, depending on capture settings).".into(),
            );
        }
        if self.gpu && self.devices.is_empty() {
            return Err("No OpenCL GPU available. Check the driver or select CPU.".into());
        }
        let input = input.clone();
        self.clear_preview();
        let mut cmd = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
        cmd.args([
            "--decode-native",
            "--backend",
            if self.gpu { "opencl" } else { "cpu" },
            "--input",
            &input,
            "--input-mode",
            if self.network {
                "tcp"
            } else if self.live {
                "cxadc"
            } else {
                "file"
            },
            "--output",
            &self.output,
            "--sample-format",
            if self.network {
                "auto"
            } else {
                &self.sample_format
            },
            "--frequency",
            &self.frequency.to_string(),
            "--start",
            &if self.live || self.network {
                0
            } else {
                self.start
            }
            .to_string(),
            "--frames",
            &self.frames.to_string(),
            "--device",
            &self.device.to_string(),
            "--luma-levels",
            if self.automatic_black {
                "automatic"
            } else {
                "reference"
            },
            "--preview-colour",
            if self.preview_colour { "true" } else { "false" },
            "--full-raster",
            if self.full_raster { "true" } else { "false" },
            "--full-raster-chroma",
            preview::RasterChroma::Whole.name(),
            "--dropout-correct",
            if self.dropout_correct {
                "true"
            } else {
                "false"
            },
        ]);
        if let Some(threshold) = self.dropout_threshold {
            cmd.args(["--dropout-threshold", &threshold.to_string()]);
        }
        if self.direct_mkv && !self.preview_only {
            cmd.args([
                "--record-mode",
                "mkv",
                "--mkv-profile",
                &self.export_profile,
                "--mkv-encoder",
                &self.export_encoder,
                "--audio-offset-ms",
                &self.audio_offset_ms.to_string(),
            ]);
        }
        if self.vbi_enabled {
            cmd.args([
                "--vbi-bind",
                &self.vbi_bind,
                "--vbi-wait",
                if self.vbi_wait { "true" } else { "false" },
                "--burn-teletext-header",
                if self.burn_teletext { "true" } else { "false" },
            ]);
            let placement = self.header_placement.fitted();
            cmd.args([
                "--teletext-header-x".into(),
                placement.x.to_string(),
                "--teletext-header-y".into(),
                placement.y.to_string(),
                "--teletext-header-size".into(),
                placement.size.to_string(),
            ]);
        }
        if self.network {
            cmd.args([
                "--audio-monitor",
                if self.audio_monitor { "true" } else { "false" },
            ]);
        }
        // Preview and recording both stream fields over a loopback connection;
        // the decoder never waits for it and the GUI shows the newest frame.
        let receiver = vhs_opencl::channel::PreviewReceiver::bind()
            .map_err(|e| format!("Preview channel: {e}"))?;
        if self.preview_only {
            cmd.args(["--preview-only", "true"]);
        }
        cmd.arg("--preview-port")
            .arg(receiver.port().map_err(|e| e.to_string())?.to_string());
        self.spawn_job(cmd, false)?;
        self.preview_socket = Some(receiver);
        self.job = Job::Decode;
        self.pending_export = (self.export_after_decode && !self.preview_only && !self.direct_mkv)
            .then(|| self.output.clone());
        self.preview_path = None;
        self.status = if self.preview_only && self.preview_colour {
            "Live PAL colour preview · no recording"
        } else if self.preview_only {
            "Live luma preview · no recording"
        } else if self.direct_mkv {
            "Direct MKV recording · no TBC files"
        } else if self.network {
            "Network RF decoding in progress"
        } else if self.live {
            "Live cxadc decoding in progress"
        } else {
            "Decoding in progress"
        }
        .into();
        Ok(())
    }
    fn launch_export(&mut self, base: &str, output: &str) -> Result<(), String> {
        if PathBuf::from(output).exists() {
            return Err("The MKV output already exists. Choose another name.".into());
        }
        let mut cmd = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
        cmd.args([
            "--export-mkv",
            "--input",
            base,
            "--output",
            output,
            "--backend",
            if self.gpu { "opencl" } else { "cpu" },
            "--device",
            &self.device.to_string(),
            "--encoder",
            &self.export_encoder,
            "--profile",
            &self.export_profile,
            "--audio-offset-ms",
            &self.audio_offset_ms.to_string(),
        ]);
        self.spawn_job(cmd, true)?;
        self.preview_only = false;
        self.pending_export = None;
        self.job = Job::Export;
        self.status = "Exporting colour MKV…".into();
        Ok(())
    }
    fn launch_server(&mut self) -> Result<(), String> {
        if !vhs_opencl::cxadc::is_stream(std::path::Path::new(&self.live_input)) {
            return Err(format!(
                "Select a cxadc device, for example {}",
                vhs_opencl::cxadc::DEFAULT_DEVICE
            ));
        }
        let mut cmd = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
        cmd.args([
            "--stream-rf",
            "--input",
            &self.live_input,
            "--bind",
            &self.server_bind,
            "--persistent",
            "true",
        ]);
        if self.sample_format != "auto" {
            cmd.args(["--sample-format", &self.sample_format]);
        }
        cmd.args(["--frequency", &self.frequency.to_string()]);
        if self.server_audio {
            cmd.args([
                "--audio-source",
                &self.audio_source,
                "--audio-bind",
                &self.audio_bind,
            ]);
        }
        self.clear_preview();
        self.spawn_job(cmd, false)?;
        self.job = Job::Server;
        self.pending_export = None;
        self.status = "RF server starting · waiting for receiver".into();
        Ok(())
    }
    fn spawn_job(&mut self, mut cmd: Command, keep_log: bool) -> Result<(), String> {
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env(vhs_opencl::native::STDIN_CONTROL, "1")
            .env(
                vhs_opencl::cxadc::BUFFER_ENV,
                self.live_buffer_mib.to_string(),
            );
        let mut child = cmd.spawn().map_err(|e| format!("Launch: {e}"))?;
        self.child_stdin = child.stdin.take();
        let (tx, rx) = mpsc::channel();
        let out = child.stdout.take().unwrap();
        let err = child.stderr.take().unwrap();
        let tx2 = tx.clone();
        // One log entry per terminal line, so the log can be drawn row by row.
        let forward = |reader, tx: mpsc::Sender<String>| {
            move || {
                for line in BufReader::new(reader).split(b'\n').map_while(Result::ok) {
                    for part in String::from_utf8_lossy(&line).split('\r') {
                        if !part.is_empty() {
                            let _ = tx.send(part.to_owned());
                        }
                    }
                }
            }
        };
        std::thread::spawn(forward(Box::new(out) as Box<dyn std::io::Read + Send>, tx));
        std::thread::spawn(forward(Box::new(err) as Box<dyn std::io::Read + Send>, tx2));
        self.child = Some(child);
        self.rx = Some(rx);
        if !keep_log {
            self.log.clear();
        }
        self.progress = None;
        self.started = Some(Instant::now());
        self.stopping = false;
        Ok(())
    }
    fn clear_preview(&mut self) {
        if let Some(path) = self.preview_path.take() {
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(path.with_extension("tmp"));
        }
        self.preview_texture = None;
        self.header_texture = None;
        self.preview_socket = None;
        self.frame_assembler.clear();
        self.metrics = None;
        self.rf_queue_s = None;
        #[cfg(feature = "ui-screenshot")]
        {
            self.preview_uploads = 0;
        }
        self.preview_field = 0;
        self.preview_error = None;
    }
    fn refresh_preview(&mut self, ctx: &egui::Context) {
        let frame = if let Some(socket) = &mut self.preview_socket {
            let mut latest = None;
            let packets = match socket.receive() {
                Ok(packets) => packets,
                Err(e) => {
                    self.preview_error = Some(e.to_string());
                    Vec::new()
                }
            };
            // A slow renderer discards stale display packets, never RF samples.
            for packet in packets {
                if let Some(header) = preview::HeaderOverlay::parse(&packet) {
                    let image = egui::ColorImage::from_rgba_unmultiplied(
                        [header.width, header.height],
                        &header.rgba,
                    );
                    match &mut self.header_texture {
                        Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
                        None => {
                            self.header_texture = Some(ctx.load_texture(
                                "teletext-header",
                                image,
                                egui::TextureOptions::LINEAR,
                            ))
                        }
                    }
                    continue;
                }
                match preview::PreviewFrame::parse(&packet) {
                    Ok(field) => {
                        if let Some(frame) = self.frame_assembler.push(field) {
                            latest = Some(Ok(frame));
                        }
                    }
                    Err(e) => latest = Some(Err(e)),
                }
            }
            let Some(frame) = latest else {
                return;
            };
            frame
        } else {
            let Some(path) = &self.preview_path else {
                return;
            };
            if !path.exists() {
                return;
            }
            preview::PreviewFrame::read(path)
        };
        match frame {
            Ok(frame) if frame.field != self.preview_field => {
                #[cfg(feature = "ui-screenshot")]
                {
                    self.preview_uploads += 1;
                }
                let image = if frame.colour {
                    egui::ColorImage::from_rgb([frame.width, frame.height], &frame.pixels)
                } else {
                    egui::ColorImage::from_gray([frame.width, frame.height], &frame.pixels)
                };
                if let Some(texture) = &mut self.preview_texture {
                    texture.set(image, egui::TextureOptions::LINEAR);
                } else {
                    self.preview_texture =
                        Some(ctx.load_texture("live-preview", image, egui::TextureOptions::LINEAR));
                }
                self.preview_field = frame.field;
                self.preview_error = None;
            }
            Ok(_) => {}
            Err(error) => self.preview_error = Some(error),
        }
    }
    fn stop(&mut self) {
        if self.child.is_some() {
            // Every job finalizes complete output when it reads "stop".
            if let Some(stdin) = &mut self.child_stdin {
                let _ = stdin.write_all(b"stop\n").and_then(|()| stdin.flush());
            }
            self.stopping = true;
            self.status = if self.job == Job::Server {
                "Stopping RF server…"
            } else if self.preview_only && self.job == Job::Decode {
                "Stopping live preview…"
            } else if self.job == Job::Export {
                "Stopping and finalizing partial MKV…"
            } else if self.direct_mkv {
                "Finalizing MKV and adding recorded audio…"
            } else {
                "Stopping and saving metadata…"
            }
            .into();
        }
    }
}
impl Drop for App {
    fn drop(&mut self) {
        self.clear_preview();
        // Only the GUI loads settings; other App values (tests) never save.
        if !self.saved_settings.is_null() {
            let current = self.settings_json();
            if current != self.saved_settings {
                let _ = settings::save(&current);
            }
        }
    }
}
/// Last used settings, restored at the next start.
impl App {
    fn settings_json(&self) -> serde_json::Value {
        serde_json::json!({
            "version": 1,
            "source": if self.server { "server" } else if self.network { "network" } else if self.live { "live" } else { "file" },
            "input": self.input,
            "network_input": self.network_input,
            "server_bind": self.server_bind,
            "live_input": self.live_input,
            "output": self.output,
            "frequency_mhz": self.frequency,
            "live_buffer_mib": self.live_buffer_mib,
            "frames": self.frames,
            "start": self.start,
            "sample_format": self.sample_format,
            "gpu": self.gpu,
            "gpu_device": self.devices.get(self.device),
            "preview_only": self.preview_only,
            "direct_mkv": self.direct_mkv,
            "export_after_decode": self.export_after_decode,
            "mkv_profile": self.export_profile,
            "mkv_encoder": self.export_encoder,
            "audio_offset_ms": self.audio_offset_ms,
            "vbi_enabled": self.vbi_enabled,
            "vbi_bind": self.vbi_bind,
            "vbi_wait": self.vbi_wait,
            "dropout_correct": self.dropout_correct,
            "dropout_threshold": self.dropout_threshold,
            "server_audio": self.server_audio,
            "audio_source": self.audio_source,
            "audio_bind": self.audio_bind,
            "audio_monitor": self.audio_monitor,
            "preview_colour": self.preview_colour,
            "full_raster": self.full_raster,
            "burn_teletext_header": self.burn_teletext,
            "teletext_header_x": self.header_placement.x,
            "teletext_header_y": self.header_placement.y,
            "teletext_header_size": self.header_placement.size,
            "automatic_black": self.automatic_black,
            "log_open": self.log_open,
            "window_width": self.window_size.map(|s| s[0]),
            "window_height": self.window_size.map(|s| s[1]),
            "window_maximized": self.window_maximized,
        })
    }
    fn apply_settings(&mut self, saved: &serde_json::Map<String, serde_json::Value>) {
        let read = settings::Reader(saved);
        match saved.get("source").and_then(|v| v.as_str()) {
            Some("server") => (self.server, self.live, self.network) = (true, true, false),
            Some("network") => (self.server, self.live, self.network) = (false, false, true),
            Some("live") => (self.server, self.live, self.network) = (false, true, false),
            Some("file") => (self.server, self.live, self.network) = (false, false, false),
            _ => {}
        }
        if !vhs_opencl::cxadc::SUPPORTED && self.live {
            (self.server, self.live) = (false, false);
        }
        read.string("input", &mut self.input);
        read.string("network_input", &mut self.network_input);
        read.string("server_bind", &mut self.server_bind);
        read.string("live_input", &mut self.live_input);
        read.string("output", &mut self.output);
        read.number("frequency_mhz", 1.0..=100.0, &mut self.frequency);
        read.count("frames", &mut self.frames);
        let mut buffer = self.live_buffer_mib;
        read.count("live_buffer_mib", &mut buffer);
        if buffer as usize >= vhs_opencl::cxadc::MIN_BUFFER_MIB {
            self.live_buffer_mib = buffer;
        }
        read.count("start", &mut self.start);
        read.choice(
            "sample_format",
            &["auto", "u8", "u16", "s16", "s8", "f32"],
            &mut self.sample_format,
        );
        read.flag("gpu", &mut self.gpu);
        if let Some(index) = saved
            .get("gpu_device")
            .and_then(|v| v.as_str())
            .and_then(|name| self.devices.iter().position(|d| d == name))
        {
            self.device = index;
        }
        read.flag("preview_only", &mut self.preview_only);
        read.flag("direct_mkv", &mut self.direct_mkv);
        read.flag("export_after_decode", &mut self.export_after_decode);
        read.choice(
            "mkv_profile",
            &["archive", "viewing"],
            &mut self.export_profile,
        );
        read.choice(
            "mkv_encoder",
            &["auto", "gpu", "cpu"],
            &mut self.export_encoder,
        );
        read.number(
            "audio_offset_ms",
            -10000.0..=10000.0,
            &mut self.audio_offset_ms,
        );
        read.flag("vbi_enabled", &mut self.vbi_enabled);
        read.string("vbi_bind", &mut self.vbi_bind);
        read.flag("vbi_wait", &mut self.vbi_wait);
        read.flag("dropout_correct", &mut self.dropout_correct);
        let mut threshold = f64::NAN;
        read.number("dropout_threshold", 0.05..=0.8, &mut threshold);
        self.dropout_threshold = (!threshold.is_nan()).then_some(threshold as f32);
        read.flag("server_audio", &mut self.server_audio);
        read.string("audio_source", &mut self.audio_source);
        read.string("audio_bind", &mut self.audio_bind);
        read.flag("audio_monitor", &mut self.audio_monitor);
        read.flag("preview_colour", &mut self.preview_colour);
        read.flag("full_raster", &mut self.full_raster);
        read.flag("burn_teletext_header", &mut self.burn_teletext);
        let mut placement = self.header_placement;
        for (key, target) in [
            ("teletext_header_x", &mut placement.x),
            ("teletext_header_y", &mut placement.y),
            ("teletext_header_size", &mut placement.size),
        ] {
            let mut value = f64::from(*target);
            read.number(key, 0.0..=3.0, &mut value);
            *target = value as f32;
        }
        if placement.is_valid() {
            self.header_placement = placement.fitted();
        }
        read.flag("automatic_black", &mut self.automatic_black);
        read.flag("log_open", &mut self.log_open);
        let (size, maximized) = saved_window(saved);
        self.window_size = size.or(self.window_size);
        self.window_maximized = maximized;
    }
    /// Save shortly after the last change, so typing does not rewrite the file
    /// on every key press and a crash still keeps recent choices.
    fn persist_settings(&mut self) {
        if self.saved_settings.is_null() {
            return;
        }
        let current = self.settings_json();
        if current == self.saved_settings {
            self.settings_changed = None;
            return;
        }
        let changed = *self.settings_changed.get_or_insert_with(Instant::now);
        if changed.elapsed() < Duration::from_millis(500) {
            return;
        }
        if let Err(e) = settings::save(&current) {
            self.log.push(format!("Settings were not saved: {e}"));
        }
        self.saved_settings = current;
        self.settings_changed = None;
    }
}
impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        #[cfg(feature = "ui-screenshot")]
        if let Ok(path) = std::env::var("VHS_GUI_SCREENSHOT") {
            let wait_for_export = std::env::var_os("VHS_GUI_EXPORT_RF_FIXTURE").is_some()
                || std::env::var_os("VHS_GUI_MONITOR_FIXTURE").is_some()
                || std::env::var_os("VHS_GUI_NETWORK_FIXTURE").is_some();
            // VHS_GUI_SCREENSHOT_AFTER_S captures a running job after that many seconds.
            let during = std::env::var("VHS_GUI_SCREENSHOT_AFTER_S")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .is_some_and(|s| self.started.is_some_and(|t| t.elapsed().as_secs_f64() >= s));
            let ready =
                during || !wait_for_export || (self.child.is_none() && self.started.is_some());
            if ctx.cumulative_pass_nr() >= 10 && ready {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            }
            let captured = ctx.input(|i| {
                for event in &i.events {
                    if let egui::Event::Screenshot { image, .. } = event {
                        image::save_buffer(
                            &path,
                            image.as_raw(),
                            image.width() as u32,
                            image.height() as u32,
                            image::ColorType::Rgba8,
                        )
                        .expect("save UI screenshot");
                        return true;
                    }
                }
                false
            });
            if captured {
                if let Ok(log) = std::env::var("VHS_GUI_CAPTURE_LOG") {
                    let _ = std::fs::write(
                        log,
                        format!(
                            "{}\nUI_PREVIEW uploads={} last_field={}\n",
                            self.log.join("\n"),
                            self.preview_uploads,
                            self.preview_field
                        ),
                    );
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        if let Some(rx) = &self.rx {
            for line in rx.try_iter() {
                if line.starts_with("VBI_SERVER ") && line.contains("waiting_for_client=true") {
                    self.status =
                        "VBI server listening · connect the teletext receiver to start decoding"
                            .into();
                }
                if line.starts_with("VBI_SERVER connected=") {
                    self.status = "Decoding · VBI receiver connected".into();
                }
                if line.starts_with("RF_SERVER ") {
                    self.status = if line.contains("waiting_for_receiver=true") {
                        "RF server listening · waiting for receiver"
                    } else if line.contains("capture_started=true") {
                        "RF server streaming · receiver connected"
                    } else {
                        "RF server · see activity log"
                    }
                    .into();
                }
                if line.starts_with("NETWORK_RF ") {
                    for value in line.split_whitespace() {
                        if let Some(rate) =
                            value.strip_prefix("rate_mhz=").and_then(|v| v.parse().ok())
                        {
                            self.frequency = rate;
                        }
                        if let Some(format) = value.strip_prefix("format=") {
                            self.sample_format = format.into();
                        }
                    }
                }
                if line.starts_with("LIVE_RF ") {
                    self.rf_queue_s = line
                        .split_whitespace()
                        .find_map(|v| v.strip_prefix("queue_ms="))
                        .and_then(|v| v.parse::<f64>().ok())
                        .map(|ms| ms / 1000.0);
                }
                if line.starts_with("PREVIEW_RATE ") {
                    let value = |key: &str| {
                        line.split_whitespace()
                            .find_map(|v| v.strip_prefix(key))
                            .and_then(|v| v.parse::<f64>().ok())
                    };
                    if let (Some(fps), Some(headroom)) =
                        (value("frames_per_second="), value("processing_realtime="))
                    {
                        self.metrics = Some((fps, headroom));
                    }
                }
                // Per-field progress arrives 50 times per second; show only the
                // latest instead of filling the log with it.
                if line.starts_with("Frame ") {
                    self.progress = Some(line);
                    continue;
                }
                if [
                    "DIRECT_MKV_PROGRESS ",
                    "DROPOUTS ",
                    "LIVE_RF ",
                    "PREVIEW_RATE ",
                ]
                .iter()
                .any(|prefix| line.starts_with(prefix))
                {
                    continue;
                }
                self.log.push(line);
            }
        }
        if self.log.len() > 1500 {
            self.log.drain(..self.log.len() - 1500);
        }
        let mut finished = None;
        if let Some(child) = &mut self.child {
            if let Ok(Some(exit)) = child.try_wait() {
                if let Some(rx) = &self.rx {
                    self.log.extend(rx.try_iter());
                }
                self.status = if self.stopping {
                    if exit.success() {
                        if self.job == Job::Server {
                            "RF server stopped".into()
                        } else if self.preview_only && self.job == Job::Decode {
                            "Preview stopped · nothing recorded".into()
                        } else {
                            "Stopped; output is partial".into()
                        }
                    } else {
                        format!("Stopped with an error ({exit}). See the log.")
                    }
                } else if exit.success() {
                    if self.job == Job::Server {
                        "RF server stopped".into()
                    } else if self.job == Job::Export {
                        "Done · colour MKV saved".into()
                    } else if self.preview_only {
                        "Preview finished · nothing recorded".into()
                    } else if self.direct_mkv {
                        "Done · direct MKV finalized".into()
                    } else {
                        "Done · TBC, chroma and JSON verified".into()
                    }
                } else {
                    format!(
                        "{} failed ({exit}). See the log.",
                        if self.job == Job::Export {
                            "Export"
                        } else {
                            "Decoding"
                        }
                    )
                };
                finished = Some(exit.success());
                if !exit.success() {
                    self.log_open = true;
                }
            }
        }
        if let Some(success) = finished {
            self.child = None;
            self.child_stdin = None;
            self.refresh_preview(ctx);
            if self.job == Job::Decode && success {
                if let Some(base) = self.pending_export.take() {
                    if let Err(e) = self.launch_export(&base, &format!("{base}.mkv")) {
                        self.status = e;
                    }
                }
            } else {
                self.pending_export = None;
            }
        }
        let running = self.child.is_some();
        if running && self.preview_checked.elapsed() >= Duration::from_millis(16) {
            self.refresh_preview(ctx);
            self.preview_checked = Instant::now();
        }
        if ctx.input(|i| i.viewport().close_requested()) && running {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.pending_export = None;
            if !self.stopping {
                self.stop();
            }
        }
        if !running {
            let dropped = ctx.input(|i| i.raw.dropped_files.first().and_then(|f| f.path.clone()));
            if let Some(p) = dropped {
                self.choose_input(p);
            }
        }
        // Opened once the main window is up, as from the About button.
        #[cfg(feature = "ui-screenshot")]
        if ctx.cumulative_pass_nr() == 20 && std::env::var_os("VHS_GUI_ABOUT_DIALOG").is_some() {
            self.show_about = true;
        }
        self.top_bar(ctx);
        self.settings_dialog(ctx, running);
        self.about_dialog(ctx);
        self.status_bar(ctx, running);
        // Wayland does not report the window rectangle; the drawable area in
        // points is the same inner size and is always known.
        if let Some(maximized) = ctx.input(|i| i.viewport().maximized) {
            self.window_maximized = maximized;
        }
        if !self.window_maximized {
            let window = ctx.screen_rect();
            self.window_size = Some([window.width().round(), window.height().round()]);
        }
        // A fixed settings column; the picture takes everything else.
        egui::SidePanel::left("settings-panel")
            .exact_width(SETTINGS_WIDTH)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::BACKGROUND)
                    .inner_margin(egui::Margin::symmetric(16, 14)),
            )
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("capture-settings-scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.add_enabled_ui(!running, |ui| self.settings(ui));
                    });
            });
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(theme::BACKGROUND)
                    .inner_margin(egui::Margin::symmetric(18, 14)),
            )
            .show(ctx, |ui| {
                // A window too small for everything scrolls instead of
                // pushing the picture or the controls out of view.
                egui::ScrollArea::vertical()
                    .id_salt("workspace-scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.workspace(ui, running));
            });
        // 25 frames per second matches PAL; repainting faster would only take
        // GPU time from OpenCL decoding.
        self.persist_settings();
        ctx.request_repaint_after(Duration::from_millis(if running { 40 } else { 100 }));
    }
}
/// Window layout: title bar, settings column, preview workspace and status bar.
impl App {
    /// Application-wide settings that rarely change.
    fn settings_dialog(&mut self, ctx: &egui::Context, running: bool) {
        if !self.show_settings {
            return;
        }
        let modal = egui::Modal::new(egui::Id::new("settings-dialog")).show(ctx, |ui| {
            ui.set_width(460.0);
            ui.label(theme::title("Settings", 20.0));
            ui.add_space(10.0);
            ui.label(theme::title("RF buffer", 15.0));
            const MIB: f64 = 1024.0 * 1024.0;
            let installed = vhs_opencl::cxadc::installed_memory();
            // Leave at least a quarter of the installed memory to everything else.
            let largest_mib = installed.map_or(65_536.0, |bytes| bytes as f64 * 0.75 / MIB);
            let mut gib = self.live_buffer_mib as f64 / 1024.0;
            theme::row(ui, "RAM limit", |ui| {
                ui.spacing_mut().slider_width = 200.0;
                ui.add(
                    egui::Slider::new(&mut gib, 0.25..=(largest_mib / 1024.0).max(0.25))
                        .step_by(0.25)
                        .fixed_decimals(2)
                        .suffix(" GiB"),
                )
            });
            self.live_buffer_mib =
                ((gib * 1024.0).round() as u32).max(vhs_opencl::cxadc::MIN_BUFFER_MIB as u32);
            let bytes_per_second = self.frequency
                * 1e6
                * if self.sample_format == "u16" {
                    2.0
                } else {
                    1.0
                };
            theme::hint(
                ui,
                format!(
                    "Holds RF from live cxadc capture and Network RF while decoding or output \
                     falls behind; memory is only used while there is a backlog. At {:.3} MHz \
                     this absorbs about {:.0} s of lag before capture stops with an overflow{}.",
                    self.frequency,
                    self.live_buffer_mib as f64 * MIB / bytes_per_second,
                    installed.map_or(String::new(), |bytes| format!(
                        " · {:.1} GiB installed",
                        bytes as f64 / (1024.0 * MIB)
                    )),
                ),
            );
            if running {
                theme::hint(
                    ui,
                    "The running job keeps its limit; changes apply to the next one.",
                );
            }
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                if ui.add(theme::secondary_button("Default")).clicked() {
                    self.live_buffer_mib = vhs_opencl::cxadc::DEFAULT_BUFFER_MIB as u32;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(theme::primary_button("Close")).clicked() {
                        self.show_settings = false;
                    }
                });
            });
        });
        if modal.should_close() {
            self.show_settings = false;
        }
    }
    /// A separate window laid out like the TeletextRecoveReese About window:
    /// the picture with the name and version, the author line, then project
    /// links and Close.
    fn about_dialog(&mut self, ctx: &egui::Context) {
        if !self.show_about {
            return;
        }
        const PICTURE: egui::Vec2 = egui::vec2(401.0, 500.0);
        if self.about_texture.is_none() {
            self.about_texture = ABOUT_IMAGE
                .and_then(|bytes| image::load_from_memory(bytes).ok())
                .map(|picture| {
                    let picture = picture.to_rgba8();
                    let size = [picture.width() as usize, picture.height() as usize];
                    ctx.load_texture(
                        "about-picture",
                        egui::ColorImage::from_rgba_unmultiplied(size, picture.as_raw()),
                        egui::TextureOptions::LINEAR,
                    )
                });
        }
        let window = egui::ViewportBuilder::default()
            .with_title("About Oxide RF")
            .with_app_id("oxide-rf")
            .with_inner_size([PICTURE.x + 32.0, PICTURE.y + 122.0])
            .with_resizable(false)
            .with_minimize_button(false)
            .with_maximize_button(false)
            .with_window_level(egui::WindowLevel::AlwaysOnTop);
        let id = egui::ViewportId::from_hash_of("about-window");
        // Like a modal dialog: the main window is dimmed and ignores input
        // until About is closed; clicking it brings About back to the front.
        // Skipped where About is drawn inside the main window.
        let blocker = (!ctx.embed_viewports()).then(|| {
            egui::Area::new(egui::Id::new("about-blocker"))
                .order(egui::Order::Foreground)
                .fixed_pos(egui::Pos2::ZERO)
                .show(ctx, |ui| {
                    let screen = ui.ctx().screen_rect();
                    ui.painter()
                        .rect_filled(screen, 0.0, egui::Color32::from_black_alpha(140));
                    ui.allocate_rect(screen, egui::Sense::click_and_drag())
                })
        });
        if blocker.is_some_and(|area| area.inner.clicked()) {
            ctx.send_viewport_cmd_to(id, egui::ViewportCommand::Focus);
        }
        ctx.show_viewport_immediate(id, window, |ctx, class| {
            if ctx.cumulative_pass_nr() < 3 {
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
            if class == egui::ViewportClass::Embedded {
                // Backends without native child windows.
                egui::Window::new("About Oxide RF")
                    .collapsible(false)
                    .resizable(false)
                    .show(ctx, |ui| self.about_contents(ui, PICTURE));
            } else {
                egui::CentralPanel::default()
                    .frame(egui::Frame::new().fill(theme::PANEL).inner_margin(16))
                    .show(ctx, |ui| self.about_contents(ui, PICTURE));
                if ctx.input(|i| i.viewport().close_requested() || i.key_pressed(egui::Key::Escape))
                {
                    self.show_about = false;
                }
            }
        });
    }
    fn about_contents(&mut self, ui: &mut egui::Ui, picture: egui::Vec2) {
        ui.set_width(picture.x);
        ui.vertical_centered(|ui| {
            let (area, _) = ui.allocate_exact_size(picture, egui::Sense::hover());
            let painter = ui.painter_at(area);
            match &self.about_texture {
                Some(texture) => {
                    egui::Image::new(texture).paint_at(ui, area);
                }
                None => {
                    painter.rect_filled(area, 0.0, egui::Color32::BLACK);
                }
            }
            // Name and version over the top of the picture, with a shadow so
            // they read on any background.
            let lines = [
                (
                    "Oxide RF",
                    egui::FontId::new(44.0, egui::FontFamily::Name(theme::MEDIUM.into())),
                    egui::Color32::WHITE,
                    18.0,
                ),
                (
                    concat!("v", env!("CARGO_PKG_VERSION")),
                    egui::FontId::new(18.0, egui::FontFamily::Name(theme::MEDIUM.into())),
                    egui::Color32::from_gray(225),
                    70.0,
                ),
            ];
            for (text, font, colour, top) in lines {
                let at = area.center_top() + egui::vec2(0.0, top);
                for offset in [egui::vec2(2.0, 2.0), egui::vec2(1.0, 1.0)] {
                    painter.text(
                        at + offset,
                        egui::Align2::CENTER_TOP,
                        text,
                        font.clone(),
                        egui::Color32::from_black_alpha(170),
                    );
                }
                painter.text(at, egui::Align2::CENTER_TOP, text, font, colour);
            }
            ui.add_space(10.0);
            ui.label("Created by: Siniša Došen  © 2026");
        });
        ui.add_space(10.0);
        let link = |ui: &mut egui::Ui, icon: egui::RichText, text: &str, url: &str| {
            let mut job = egui::text::LayoutJob::default();
            icon.append_to(
                &mut job,
                ui.style(),
                egui::FontSelection::Default,
                egui::Align::Center,
            );
            egui::RichText::new(format!("  {text}")).append_to(
                &mut job,
                ui.style(),
                egui::FontSelection::Default,
                egui::Align::Center,
            );
            if ui
                .add(egui::Button::new(job).min_size(egui::vec2(0.0, 32.0)))
                .on_hover_text(url)
                .clicked()
            {
                ui.ctx().open_url(egui::OpenUrl::new_tab(url));
            }
        };
        ui.horizontal(|ui| {
            link(
                ui,
                egui::RichText::new("{ Rust }")
                    .monospace()
                    .strong()
                    .color(egui::Color32::from_rgb(0xDE, 0xA5, 0x84)),
                "GitHub",
                "https://github.com/sdosenn/OxideRF",
            );
            link(
                ui,
                egui::RichText::new("☕").size(16.0),
                "Ko-fi",
                "https://ko-fi.com/sinisinavideoteka",
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add(egui::Button::new("Close").min_size(egui::vec2(80.0, 32.0)))
                    .clicked()
                {
                    self.show_about = false;
                }
            });
        });
    }
    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("title-bar")
            .frame(
                egui::Frame::new()
                    .fill(theme::PANEL)
                    .inner_margin(egui::Margin::symmetric(20, 12)),
            )
            .show(ctx, |ui| {
                // The name and version are in the window title.
                ui.horizontal(|ui| {
                    if ui.add(theme::secondary_button("Settings")).clicked() {
                        self.show_settings = true;
                    }
                    if ui.add(theme::secondary_button("About")).clicked() {
                        self.show_about = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let device = if self.gpu {
                            self.devices
                                .get(self.device)
                                .map(String::as_str)
                                .unwrap_or("No OpenCL GPU")
                        } else {
                            "CPU processing"
                        };
                        ui.label(egui::RichText::new(device).color(theme::MUTED));
                    });
                });
            });
    }
    fn state(&self, running: bool) -> (&'static str, egui::Color32) {
        if !running {
            ("Idle", theme::MUTED)
        } else if self.stopping {
            ("Finishing", theme::WARNING)
        } else if self.job == Job::Server {
            ("Serving RF", theme::ACCENT)
        } else if self.job == Job::Export {
            ("Exporting", theme::ACCENT)
        } else if self.preview_only {
            ("Preview", theme::ACCENT)
        } else {
            ("Recording", theme::RECORD)
        }
    }
    fn status_bar(&self, ctx: &egui::Context, running: bool) {
        egui::TopBottomPanel::bottom("status-bar")
            .frame(
                egui::Frame::new()
                    .fill(theme::PANEL)
                    .inner_margin(egui::Margin::symmetric(20, 8)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let (state, colour) = self.state(running);
                    theme::pill(ui, state, colour);
                    ui.label(&self.status);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        theme::hint(
                            ui,
                            if self.server {
                                "RF server · no local recording or GPU processing"
                            } else if self.preview_only {
                                "Preview only · no output files"
                            } else if self.direct_mkv {
                                if self.full_raster {
                                    "Output: name.mkv · full raster 1136 × 626 FFV1 · audio added on Stop"
                                } else {
                                    "Output: name.mkv · audio added on Stop"
                                }
                            } else {
                                "Output: name.tbc + name_chroma.tbc + name.tbc.json"
                            },
                        );
                    });
                });
            });
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        theme::card(ui, "Source", |ui| self.source_settings(ui));
        if !self.server {
            ui.add_space(12.0);
            theme::card(ui, "Recording", |ui| self.recording_settings(ui));
        }
        ui.add_space(12.0);
        theme::card(ui, "RF signal", |ui| self.signal_settings(ui));
        if !self.server {
            ui.add_space(12.0);
            theme::card(ui, "Processing", |ui| self.processing_settings(ui));
            ui.add_space(12.0);
            theme::card(ui, "MKV", |ui| self.mkv_settings(ui));
            ui.add_space(12.0);
            theme::card(ui, "Teletext / VBI", |ui| self.vbi_settings(ui));
        }
        ui.add_space(8.0);
    }
    fn source_settings(&mut self, ui: &mut egui::Ui) {
        let current = if self.server {
            3
        } else if self.network {
            2
        } else if self.live {
            1
        } else {
            0
        };
        // cxadc capture and the RF server need cxadc (Linux) or cxadc-win (Windows).
        let choice = if vhs_opencl::cxadc::SUPPORTED {
            theme::segmented(
                ui,
                current,
                &["RF file", "Live cxadc", "Network RF", "RF server"],
            )
        } else {
            theme::segmented(ui, current / 2, &["RF file", "Network RF"]).map(|c| c * 2)
        };
        if let Some(choice) = choice {
            let was_live = self.live;
            match choice {
                0 => {
                    self.live = false;
                    self.network = false;
                    self.server = false;
                }
                1 => {
                    self.live = true;
                    self.network = false;
                    self.server = false;
                }
                2 => {
                    self.network = true;
                    self.live = false;
                    self.server = false;
                    self.preview_only = true;
                    self.start = 0;
                    self.frames = 0;
                }
                _ => {
                    self.server = true;
                    self.live = true;
                    self.network = false;
                    self.preview_only = true;
                    self.refresh_cxadc(true);
                }
            }
            if self.live && !was_live {
                self.preview_only = true;
                self.start = 0;
                self.frames = 0;
                self.sample_format = "u8".into();
                self.refresh_cxadc(true);
            }
        }
        ui.add_space(6.0);
        if self.server {
            theme::row(ui, "Listen address", |ui| {
                theme::text_field(ui, &mut self.server_bind)
            });
            theme::hint(
                ui,
                "Serves raw RF to one receiver on a trusted LAN. Connect with Network RF on the \
                 viewing PC; the server stays available after a disconnect.",
            );
            ui.checkbox(&mut self.server_audio, "Send VCR audio");
            if self.server_audio {
                theme::row(ui, "Audio input", |ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Refresh").clicked() {
                            self.audio_devices = vhs_opencl::audio::devices();
                        }
                        egui::ComboBox::from_id_salt("audio-input")
                            .width(ui.available_width() - 2.0)
                            .selected_text(
                                self.audio_devices
                                    .iter()
                                    .find(|d| d.name == self.audio_source)
                                    .map(|d| d.description.as_str())
                                    .unwrap_or(&self.audio_source),
                            )
                            .show_ui(ui, |ui| {
                                for device in &self.audio_devices {
                                    ui.selectable_value(
                                        &mut self.audio_source,
                                        device.name.clone(),
                                        &device.description,
                                    );
                                }
                            });
                    });
                });
                theme::row(ui, "Source name", |ui| {
                    theme::text_field(ui, &mut self.audio_source)
                });
                theme::row(ui, "Audio address", |ui| {
                    theme::text_field(ui, &mut self.audio_bind)
                });
                theme::hint(
                    ui,
                    if cfg!(target_os = "linux") {
                        "48 kHz stereo PCM via PulseAudio/PipeWire-Pulse · monitored and recorded on the client."
                    } else {
                        "48 kHz stereo PCM from the selected input · monitored and recorded on the client."
                    },
                );
            }
        } else if self.network {
            theme::row(ui, "RF server", |ui| {
                theme::text_field(ui, &mut self.network_input)
            });
            theme::hint(
                ui,
                "Format and sample rate come from the capture PC. 28.636 MHz u8 needs about \
                 229 Mbit/s on a wired LAN.",
            );
            ui.checkbox(&mut self.audio_monitor, "Listen to server audio");
            theme::hint(
                ui,
                "Server audio is recorded as .audio.wav with timing JSON and added to the MKV.",
            );
        } else if self.live {
            if theme::field_with_button(ui, "Device", &mut self.live_input, "Refresh") {
                self.refresh_cxadc(false);
            }
            theme::row(ui, "Detected", |ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Use its format").clicked() {
                        self.apply_cxadc_settings();
                    }
                    egui::ComboBox::from_id_salt("cxadc-device")
                        .width(ui.available_width() - 2.0)
                        .selected_text(if self.cxadc_devices.is_empty() {
                            "No cxadc devices"
                        } else {
                            "Choose device"
                        })
                        .show_ui(ui, |ui| {
                            for device in &self.cxadc_devices {
                                ui.selectable_value(
                                    &mut self.live_input,
                                    device.path.to_string_lossy().into_owned(),
                                    device.path.display().to_string(),
                                );
                            }
                            if self.cxadc_devices.is_empty() {
                                ui.label("No cxadc devices detected");
                            }
                        });
                });
            });
            if let Some(device) = self
                .cxadc_devices
                .iter()
                .find(|d| d.path == std::path::Path::new(&self.live_input))
            {
                theme::hint(ui, device.summary());
            }
            theme::hint(ui, "Direct RF stream with a bounded capture buffer.");
        } else {
            if theme::field_with_button(ui, "RF capture", &mut self.input, "Open…") {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter(
                        "RF captures",
                        &[
                            "raw", "r8", "u8", "s8", "r16", "u16", "s16", "rf", "flac", "wav",
                            "ldf", "vhs",
                        ],
                    )
                    .pick_file()
                {
                    self.choose_input(p);
                }
            }
        }
    }
    fn recording_settings(&mut self, ui: &mut egui::Ui) {
        let mode = if self.preview_only {
            0
        } else if self.direct_mkv {
            2
        } else {
            1
        };
        if let Some(choice) =
            theme::segmented(ui, mode, &["Preview only", "TBC files", "Direct MKV"])
        {
            self.preview_only = choice == 0;
            if choice > 0 {
                self.direct_mkv = choice == 2;
            }
        }
        theme::hint(
            ui,
            if self.preview_only {
                "Live picture only · nothing is written to disk."
            } else if self.direct_mkv {
                if self.full_raster {
                    "Full raster: every TBC sample and line (1136 × 626), lossless FFV1 10-bit \
                     4:2:2. Audio is added on Stop."
                } else {
                    "Encodes video as fields arrive; recorded audio is added on Stop without \
                     re-encoding."
                }
            } else {
                "Writes name.tbc, name_chroma.tbc and name.tbc.json for later processing."
            },
        );
        ui.add_space(4.0);
        ui.add_enabled_ui(!self.preview_only, |ui| {
            if theme::field_with_button(ui, "Output name", &mut self.output, "Save as…") {
                if let Some(p) = rfd::FileDialog::new()
                    .set_file_name("vhs-decoded")
                    .save_file()
                {
                    self.output = p.to_string_lossy().into();
                }
            }
        });
        theme::row(ui, "Frame count", |ui| {
            ui.add(egui::DragValue::new(&mut self.frames));
            theme::hint(
                ui,
                if self.live || self.network {
                    "0 = until stopped"
                } else {
                    "0 = entire capture"
                },
            );
        });
        if !self.live && !self.network {
            theme::row(ui, "Start frame", |ui| {
                ui.add(egui::DragValue::new(&mut self.start))
            });
        }
    }
    fn signal_settings(&mut self, ui: &mut egui::Ui) {
        ui.add_enabled_ui(!self.network, |ui| {
            theme::row(ui, "Sample rate", |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.frequency)
                        .speed(0.001)
                        .range(1.0..=100.0)
                        .max_decimals(9)
                        .suffix(" MHz"),
                );
                // Common cxadc rates; the value stays editable for any other clock.
                let presets = [
                    (315.0 / 11.0, "28.636 MHz · 8fsc stock crystal"),
                    (40.0, "40 MHz · crystal mod / clockgen"),
                    (315.0 / 11.0 * 1.25, "35.795 MHz · 10fsc (tenxfsc=1)"),
                    (20.0, "20 MHz · clockgen"),
                    (50.0, "50 MHz · clockgen"),
                ];
                egui::ComboBox::from_id_salt("sample-rate")
                    .width(ui.available_width() - 2.0)
                    .selected_text(
                        presets
                            .iter()
                            .find(|(rate, _)| (rate - self.frequency).abs() < 1e-6)
                            // The value is beside the list; show only the name,
                            // so a narrow column is not pushed wider.
                            .map_or("Custom", |(_, label)| {
                                label.split_once(" · ").map_or(*label, |(_, name)| name)
                            }),
                    )
                    .show_ui(ui, |ui| {
                        for (rate, label) in presets {
                            ui.selectable_value(&mut self.frequency, rate, label);
                        }
                    });
            });
            theme::row(ui, "Sample format", |ui| {
                let auto = if self.network {
                    "Auto from sender"
                } else if self.live {
                    "Auto from device"
                } else {
                    "Auto by extension"
                };
                let formats = [
                    ("auto", auto),
                    ("u8", "Unsigned 8-bit (cxadc)"),
                    ("u16", "Unsigned 16-bit LE (cxadc)"),
                    ("s16", "Signed 16-bit LE"),
                    ("s8", "Signed 8-bit"),
                    ("f32", "Float 32-bit LE"),
                ];
                egui::ComboBox::from_id_salt("sample-format")
                    .width(ui.available_width() - 2.0)
                    .selected_text(
                        formats
                            .iter()
                            .find(|(key, _)| *key == self.sample_format)
                            .map_or(self.sample_format.as_str(), |(_, label)| *label),
                    )
                    .show_ui(ui, |ui| {
                        for (key, label) in formats {
                            if self.live && !["auto", "u8", "u16"].contains(&key) {
                                continue;
                            }
                            ui.selectable_value(&mut self.sample_format, key.into(), label);
                        }
                    });
            });
            theme::hint(
                ui,
                if self.network {
                    "Format and clock are supplied by the capture PC."
                } else if self.live {
                    "u8 = 8-bit RF. u16 = 16-bit LE at the device's halved sample rate."
                } else {
                    ".r8/.u8 = unsigned 8-bit, .r16/.u16 = unsigned 16-bit; choose the format for \
                     .raw. Use auto for RF FLAC/WAV."
                },
            );
        });
    }
    fn processing_settings(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.gpu, "OpenCL GPU acceleration");
        ui.add_enabled_ui(self.gpu, |ui| {
            theme::row(ui, "GPU", |ui| {
                egui::ComboBox::from_id_salt("gpu-device")
                    .width(ui.available_width() - 2.0)
                    .selected_text(
                        self.devices
                            .get(self.device)
                            .map(String::as_str)
                            .unwrap_or("No OpenCL GPU devices"),
                    )
                    .show_ui(ui, |ui| {
                        for (i, name) in self.devices.iter().enumerate() {
                            ui.selectable_value(&mut self.device, i, name);
                        }
                    });
            });
        });
        theme::hint(
            ui,
            if self.preview_only && self.preview_colour {
                "GPU: RF, sync, TBC and PAL colour. Colour uses more GPU time than luma."
            } else if self.preview_only {
                "GPU: RF, FM, sync and visible luma. Chroma and TBC recording are skipped."
            } else {
                "GPU: RF filters, FM, sync edges, TBC and chroma. CPU: input/output and field tracking."
            },
        );
        ui.add_space(4.0);
        theme::row(ui, "Luma levels", |ui| {
            if let Some(choice) = theme::segmented(
                ui,
                usize::from(self.automatic_black),
                &["VHS reference", "Automatic black"],
            ) {
                self.automatic_black = choice == 1;
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.dropout_correct, "Correct RF dropouts");
            ui.add_enabled_ui(self.dropout_correct, |ui| {
                let default = vhs_opencl::dropout::default_threshold(self.frequency);
                let mut threshold = self.dropout_threshold.unwrap_or(default);
                if ui
                    .add(
                        egui::DragValue::new(&mut threshold)
                            .speed(0.01)
                            .range(0.05..=0.8)
                            .prefix("threshold "),
                    )
                    .on_hover_text("Fraction of the field's RF level below which a dropout starts.")
                    .changed()
                {
                    self.dropout_threshold = Some(threshold);
                }
                if ui
                    .add_enabled(self.dropout_threshold.is_some(), egui::Button::new("Reset"))
                    .on_hover_text(format!(
                        "vhs-decode default for this sample rate: {default} \
                         (0.35 at ~28 MHz cxadc, 0.18 otherwise)"
                    ))
                    .clicked()
                {
                    self.dropout_threshold = None;
                }
            });
        });
        if self.dropout_correct {
            theme::hint(
                ui,
                "Conceals damaged luma and chroma. VBI/teletext and sync are preserved.",
            );
        }
    }
    fn mkv_settings(&mut self, ui: &mut egui::Ui) {
        let full = self.full_raster && self.direct_mkv && !self.preview_only;
        if full {
            theme::hint(
                ui,
                "Full-raster recording always uses lossless FFV1 10-bit 4:2:2.",
            );
        }
        ui.add_enabled_ui(!full, |ui| {
            theme::row(ui, "Profile", |ui| {
                egui::ComboBox::from_id_salt("mkv-profile")
                    .width(ui.available_width() - 2.0)
                    .selected_text(if self.export_profile == "archive" {
                        "Archive · FFV1 10-bit 4:2:2 interlaced"
                    } else {
                        "Viewing · H.264 8-bit 4:2:0 50p"
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.export_profile,
                            "archive".into(),
                            "Archive · FFV1 10-bit 4:2:2 interlaced",
                        );
                        ui.selectable_value(
                            &mut self.export_profile,
                            "viewing".into(),
                            "Viewing · H.264 8-bit 4:2:0 50p",
                        );
                    });
            });
        });
        ui.add_enabled_ui(self.export_profile == "viewing" && !full, |ui| {
            theme::row(ui, "Encoder", |ui| {
                egui::ComboBox::from_id_salt("mkv-encoder")
                    .width(ui.available_width() - 2.0)
                    .selected_text(match self.export_encoder.as_str() {
                        "gpu" => "GPU required",
                        "cpu" => "CPU H.264",
                        _ => "Automatic GPU / CPU",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.export_encoder,
                            "auto".into(),
                            "Automatic GPU / CPU",
                        );
                        ui.selectable_value(&mut self.export_encoder, "gpu".into(), "GPU required");
                        ui.selectable_value(&mut self.export_encoder, "cpu".into(), "CPU H.264");
                    });
            });
        });
        theme::row(ui, "Audio offset", |ui| {
            ui.add(
                egui::DragValue::new(&mut self.audio_offset_ms)
                    .speed(1.0)
                    .range(-10000.0..=10000.0)
                    .suffix(" ms"),
            );
            theme::hint(ui, "positive delays audio");
        });
        ui.add_enabled_ui(!self.preview_only && !self.direct_mkv, |ui| {
            ui.checkbox(
                &mut self.export_after_decode,
                "Export MKV after decoding TBC",
            );
        });
        if ui.button("Export existing TBC to MKV…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Luma TBC", &["tbc"])
                .pick_file()
            {
                let base = path.with_extension("").to_string_lossy().into_owned();
                if base.ends_with("_chroma") {
                    self.status = "Select the luma .tbc, not the _chroma.tbc file.".into();
                } else if let Some(output) = rfd::FileDialog::new()
                    .set_directory(path.parent().unwrap_or(std::path::Path::new(".")))
                    .set_file_name(format!(
                        "{}.mkv",
                        path.file_stem().unwrap_or_default().to_string_lossy()
                    ))
                    .add_filter("Matroska video", &["mkv"])
                    .save_file()
                {
                    if let Err(e) = self.launch_export(&base, &output.to_string_lossy()) {
                        self.status = e;
                    }
                }
            }
        }
        theme::hint(
            ui,
            if self.export_profile == "archive" {
                "720 × 576 · PAL 25i · 4:3 · colour · CPU FFV1 · system FFmpeg required"
            } else {
                "720 × 576 · PAL 50p · colour · GPU or CPU H.264 · system FFmpeg required"
            },
        );
    }
    fn vbi_settings(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.vbi_enabled, "Stream VBI over TCP");
        ui.add_enabled_ui(self.vbi_enabled, |ui| {
            theme::row(ui, "Listen address", |ui| {
                theme::text_field(ui, &mut self.vbi_bind)
            });
            ui.checkbox(
                &mut self.vbi_wait,
                "Wait for teletext receiver before decoding",
            );
            ui.checkbox(&mut self.burn_teletext, "Burn teletext header into the MKV");
        });
        theme::hint(
            ui,
            "VBI-only TBC · 1135 samples × 16 lines · UInt16 LE · raw TCP, no header.",
        );
    }

    fn workspace(&mut self, ui: &mut egui::Ui, running: bool) {
        let top = ui.cursor().top();
        let mut picture_height = None;
        if self.server {
            theme::card(ui, "RF sender", |ui| {
                ui.label("Watch and record on the receiving PC using Network RF.");
                ui.label("One receiver at a time. The server remains available between sessions.");
                theme::hint(
                    ui,
                    "28.636 MHz u8 requires about 229 Mbit/s on a wired gigabit LAN.",
                );
            });
            ui.add_space(14.0);
            ui.horizontal(|ui| self.actions(ui, running, None));
        } else {
            // The picture is the point of the workspace: it takes the full
            // width, or the full height left by the rows under it, whichever
            // is tighter. Those rows wrap by the workspace width only, and
            // their height is measured from the layout cursor in the previous
            // frame, so the picture size never feeds back into itself.
            let available = ui.available_size();
            let aspect = self.preview_aspect();
            let by_height = available.y - 4.0 - self.workspace_chrome;
            let height = by_height.min(available.x / aspect).max(120.0);
            let picture = egui::vec2(height * aspect, height);
            picture_height = Some(height);
            ui.horizontal(|ui| {
                ui.add_space(((available.x - picture.x) / 2.0).max(0.0));
                self.preview_picture(ui, running, picture);
            });
            ui.add_space(8.0);
            // The button and the switches share a row when their measured
            // widths fit; neither depends on the picture size.
            let switches_beside = available.x >= self.toolbar_width;
            let mut widths = [0.0; 2];
            ui.horizontal(|ui| {
                widths[0] = ui
                    .scope(|ui| {
                        self.actions(ui, running, Some(egui::vec2(140.0, theme::CONTROL_HEIGHT)))
                    })
                    .response
                    .rect
                    .width();
                if switches_beside {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        widths[1] = ui
                            .scope(|ui| self.preview_switches(ui, running, true))
                            .response
                            .rect
                            .width();
                    });
                }
            });
            if !switches_beside {
                ui.horizontal(|ui| {
                    widths[1] = ui
                        .scope(|ui| self.preview_switches(ui, running, false))
                        .response
                        .rect
                        .width();
                });
            }
            self.toolbar_width = widths[0] + widths[1] + 24.0;
            let columns = if available.x >= METRICS_ONE_ROW_WIDTH {
                4
            } else {
                2
            };
            self.metrics_rows(ui, running, columns);
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    self.log_toggle(ui);
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        self.picture_info(ui, running);
                        self.picture_hints(ui);
                    });
                });
            });
            self.picture_notes(ui, running);
        }
        if self.server {
            ui.add_space(8.0);
            self.log_toggle(ui);
        }
        if self.log_open {
            egui::Frame::new()
                .fill(theme::FIELD)
                .stroke(egui::Stroke::new(1.0_f32, theme::CARD_STROKE))
                .corner_radius(egui::CornerRadius::same(10))
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    // Lay out only the visible rows; the log holds up to 1500 lines.
                    let row_height = ui.text_style_height(&egui::TextStyle::Monospace);
                    egui::ScrollArea::both()
                        .max_height(180.0)
                        .stick_to_bottom(true)
                        .auto_shrink([false, true])
                        .show_rows(ui, row_height, self.log.len(), |ui, rows| {
                            for line in &self.log[rows] {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(line).monospace().color(theme::MUTED),
                                    )
                                    .extend(),
                                );
                            }
                        });
                });
        }
        if let Some(height) = picture_height {
            // Everything but the picture, from the cursor: the scroll area
            // stretches the layout's own rectangle to the viewport.
            self.workspace_chrome = (ui.cursor().top() - top - height).max(0.0);
        }
    }
    /// The teletext header on the preview, edited like a source in OBS: drag
    /// the box to move it, drag a corner to resize it with its aspect kept,
    /// double-click to reset. The frame and handles are drawn by the GUI only;
    /// a recording burns the header image at the placement it started with.
    fn header_editor(&mut self, ui: &egui::Ui, area: egui::Rect, frame_width: usize) {
        let to_screen = |[x, y, w, h]: [f32; 4]| {
            egui::Rect::from_min_size(
                area.min + egui::vec2(x * area.width(), y * area.height()),
                egui::vec2(w * area.width(), h * area.height()),
            )
        };
        let active = to_screen(preview::HeaderOverlay::active_area(frame_width));
        self.header_placement = self.header_placement.fitted();
        let rect = to_screen(preview::HeaderOverlay::placement(
            frame_width,
            self.header_placement,
        ));
        match &self.header_texture {
            Some(header) => {
                egui::Image::new(header).paint_at(ui, rect);
            }
            None => {
                ui.painter()
                    .rect_filled(rect, 0.0, egui::Color32::from_black_alpha(140));
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Teletext header",
                    egui::FontId::proportional((rect.height() * 0.6).clamp(9.0, 18.0)),
                    theme::MUTED,
                );
            }
        }
        let placement = &mut self.header_placement;
        let body = ui.interact(
            rect,
            ui.id().with("header-move"),
            egui::Sense::click_and_drag(),
        );
        if body.dragged() {
            placement.x += body.drag_delta().x / active.width();
            placement.y += body.drag_delta().y / active.height();
        }
        if body.double_clicked() {
            *placement = preview::HeaderPlacement::default();
        }
        // Corners are added after the body so they win where they overlap.
        let mut active_handle = false;
        let mut hovered = body.hovered() || body.dragged();
        let corners = [
            (
                rect.left_top(),
                rect.right_bottom(),
                egui::CursorIcon::ResizeNwSe,
            ),
            (
                rect.right_top(),
                rect.left_bottom(),
                egui::CursorIcon::ResizeNeSw,
            ),
            (
                rect.left_bottom(),
                rect.right_top(),
                egui::CursorIcon::ResizeNeSw,
            ),
            (
                rect.right_bottom(),
                rect.left_top(),
                egui::CursorIcon::ResizeNwSe,
            ),
        ];
        for (index, (corner, anchor, cursor)) in corners.into_iter().enumerate() {
            let handle = ui.interact(
                egui::Rect::from_center_size(corner, egui::vec2(14.0, 14.0)),
                ui.id().with(("header-resize", index)),
                egui::Sense::drag(),
            );
            if handle.hovered() || handle.dragged() {
                ui.ctx().set_cursor_icon(cursor);
                hovered = true;
            }
            if let Some(pointer) = handle.interact_pointer_pos().filter(|_| handle.dragged()) {
                active_handle = true;
                // Scale so the dragged corner follows the pointer and the
                // opposite corner stays put.
                let unit = preview::HeaderPlacement {
                    size: 1.0,
                    ..*placement
                }
                .active_rect();
                let (unit_w, unit_h) = (unit[2] * active.width(), unit[3] * active.height());
                // Project the pointer onto the box diagonal, as OBS does with
                // the aspect ratio locked.
                let span = (pointer - anchor).abs();
                // The fixed corner limits growth to the picture edges.
                let room_x = if corner.x < anchor.x {
                    anchor.x - active.left()
                } else {
                    active.right() - anchor.x
                };
                let room_y = if corner.y < anchor.y {
                    anchor.y - active.top()
                } else {
                    active.bottom() - anchor.y
                };
                let (smallest, largest) = (
                    *preview::HeaderPlacement::SIZE_RANGE.start(),
                    *preview::HeaderPlacement::SIZE_RANGE.end(),
                );
                let size = ((span.x * unit_w + span.y * unit_h)
                    / (unit_w * unit_w + unit_h * unit_h))
                    .clamp(
                        smallest,
                        largest
                            .min(room_x / unit_w)
                            .min(room_y / unit_h)
                            .max(smallest),
                    );
                let (w, h) = (unit_w * size, unit_h * size);
                let left = if corner.x < anchor.x {
                    anchor.x - w
                } else {
                    anchor.x
                };
                let top = if corner.y < anchor.y {
                    anchor.y - h
                } else {
                    anchor.y
                };
                placement.size = size;
                placement.x = (left - active.left()) / active.width();
                placement.y = (top - active.top()) / active.height();
            }
        }
        // The header never leaves the picture, whether moved or resized.
        *placement = placement.fitted();
        if body.hovered() && !active_handle {
            ui.ctx().set_cursor_icon(if body.dragged() {
                egui::CursorIcon::Grabbing
            } else {
                egui::CursorIcon::Grab
            });
        }
        // Outline and handles only while the pointer is over the preview.
        if hovered || ui.rect_contains_pointer(area) {
            let stroke = egui::Stroke::new(1.5_f32, theme::ACCENT);
            ui.painter()
                .rect_stroke(rect, 0.0, stroke, egui::StrokeKind::Outside);
            for (corner, _, _) in corners {
                let handle = egui::Rect::from_center_size(corner, egui::vec2(8.0, 8.0));
                ui.painter().rect_filled(handle, 1.0, theme::ACCENT);
                ui.painter().rect_stroke(
                    handle,
                    1.0,
                    egui::Stroke::new(1.0_f32, theme::ON_ACCENT),
                    egui::StrokeKind::Inside,
                );
            }
        }
    }
    /// Display aspect of the preview: from the latest frame, else the mode.
    fn preview_aspect(&self) -> f32 {
        match &self.preview_texture {
            Some(texture) => preview::display_aspect(texture.size()[0]),
            None if self.full_raster => preview::FULL_ASPECT,
            None => 4.0 / 3.0,
        }
    }
    /// Raster and colour switches of the preview; `reversed` for a right-to-left row.
    fn preview_switches(&mut self, ui: &mut egui::Ui, running: bool, reversed: bool) {
        ui.add_enabled_ui(!running, |ui| {
            for index in if reversed { [1, 0] } else { [0, 1] } {
                if index == 0 {
                    if let Some(choice) = theme::segmented(
                        ui,
                        usize::from(self.full_raster),
                        &["4:3 picture", "Full raster"],
                    ) {
                        self.full_raster = choice == 1;
                    }
                } else if let Some(choice) = theme::segmented(
                    ui,
                    usize::from(self.preview_colour),
                    &["Luma only", "Chroma + luma"],
                ) {
                    self.preview_colour = choice == 1;
                }
            }
        });
    }
    fn log_toggle(&mut self, ui: &mut egui::Ui) {
        let label = if self.log_open {
            "Hide activity log"
        } else {
            "Show activity log"
        };
        if ui
            .add(egui::Button::new(theme::title(label, 14.0)).frame(false))
            .clicked()
        {
            self.log_open = !self.log_open;
        }
    }
    /// The decoded picture with the teletext header editor, or a placeholder.
    fn preview_picture(&mut self, ui: &mut egui::Ui, running: bool, picture: egui::Vec2) {
        let (area, _) = ui.allocate_exact_size(picture, egui::Sense::hover());
        ui.painter()
            .rect_filled(area, egui::CornerRadius::same(8), egui::Color32::BLACK);
        if let Some(texture) = &self.preview_texture {
            egui::Image::new(texture)
                .corner_radius(egui::CornerRadius::same(8))
                .paint_at(ui, area);
            // The teletext header where a TV shows row 0, from the receiver; only
            // shown when it will be burned into the MKV.
            if self.vbi_enabled && self.burn_teletext {
                let frame_width = texture.size()[0];
                self.header_editor(ui, area, frame_width);
            }
        } else {
            ui.painter().text(
                area.center(),
                egui::Align2::CENTER_CENTER,
                if running {
                    "Waiting for the first decoded field…"
                } else {
                    "The picture appears here while decoding"
                },
                egui::FontId::proportional(16.0),
                theme::MUTED,
            );
        }
    }
    /// Field position of the picture, or decoding progress while running.
    fn picture_info(&self, ui: &mut egui::Ui, running: bool) {
        if let Some(progress) = self.progress.as_ref().filter(|_| running) {
            // Per-field progress includes sync quality; show it while running.
            ui.add(
                egui::Label::new(egui::RichText::new(progress).monospace().color(theme::TEXT))
                    .truncate(),
            );
        } else if self.preview_texture.is_some() {
            ui.label(
                egui::RichText::new(format!(
                    "Frame {} · field {} · {:.2} s{} ·",
                    self.preview_field.div_ceil(2),
                    self.preview_field,
                    self.preview_field as f64 / 50.0,
                    if running { "" } else { " · last decoded" }
                ))
                .color(theme::MUTED),
            );
        }
    }
    /// One-line description of the preview mode, cut to the space left.
    fn picture_hints(&self, ui: &mut egui::Ui) {
        theme::hint_line(
            ui,
            format!(
                "{} · frames woven from adjacent fields · {}",
                if !self.full_raster {
                    "PAL 4:3"
                } else {
                    "full TBC raster 1135 × 626"
                },
                if self.preview_only && self.preview_colour {
                    "live chroma + luma, display only"
                } else if self.preview_only {
                    "live luma only, display only"
                } else if self.direct_mkv {
                    "does not change the MKV profile"
                } else if self.preview_colour {
                    "luma and chroma TBC saved separately"
                } else {
                    "chroma saved separately"
                }
            ),
        );
    }
    /// Notes that only appear in some states, each on its own line.
    fn picture_notes(&self, ui: &mut egui::Ui, running: bool) {
        if self.vbi_enabled && running && !self.preview_only && self.burn_teletext {
            theme::hint(
                ui,
                "Moving the header changes the preview only; this recording burns the \
                 position it started with.",
            );
        }
        if let Some(error) = &self.preview_error {
            ui.colored_label(theme::WARNING, error);
        }
    }
    /// One button that starts the selected job and, while it runs, stops it.
    fn actions(&mut self, ui: &mut egui::Ui, running: bool, size: Option<egui::Vec2>) {
        let fit = |button: egui::Button<'static>| match size {
            Some(size) => button.min_size(size),
            None => button,
        };
        if !running {
            let start = if self.server {
                "▶  Start server"
            } else if self.preview_only {
                "▶  Start preview"
            } else if self.live || self.network {
                "▶  Start recording"
            } else {
                "▶  Decode"
            };
            if ui.add(fit(theme::primary_button(start))).clicked() {
                if let Err(e) = self.launch() {
                    self.status = e;
                }
            }
            return;
        }
        let stop = if self.stopping {
            "■  Stopping…"
        } else if self.server {
            "■  Stop server"
        } else if self.preview_only && self.job == Job::Decode {
            "■  Stop preview"
        } else {
            "■  Stop and save"
        };
        if ui
            .add_enabled(!self.stopping, fit(theme::secondary_button(stop)))
            .clicked()
        {
            self.stop();
        }
    }
    /// Live values as fixed-width chips, `columns` to a row.
    fn metrics_rows(&self, ui: &mut egui::Ui, running: bool, columns: usize) {
        if !running && self.metrics.is_none() {
            return;
        }
        let mut tiles: Vec<(&str, String, egui::Color32)> = Vec::new();
        let mut tile = |label: &'static str, value: &str, color| {
            tiles.push((label, value.to_owned(), color));
        };
        let elapsed = self
            .started
            .filter(|_| running)
            .map(|t| t.elapsed().as_secs());
        tile(
            "Elapsed",
            &elapsed.map_or("—".into(), |s| {
                format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
            }),
            theme::TEXT,
        );
        if let Some((fps, headroom)) = self.metrics {
            tile(
                "Decode",
                &format!("{fps:.1} fps"),
                if fps >= 24.0 {
                    theme::TEXT
                } else {
                    theme::WARNING
                },
            );
            tile(
                "Headroom",
                &format!("{headroom:.2}×"),
                if headroom >= 1.3 {
                    theme::SUCCESS
                } else if headroom >= 1.0 {
                    theme::WARNING
                } else {
                    theme::DANGER
                },
            );
        }
        if let Some(backlog) = self.rf_queue_s {
            tile(
                "RF backlog",
                &format!("{backlog:.1} s"),
                if backlog < 1.0 {
                    theme::SUCCESS
                } else if backlog < 10.0 {
                    theme::WARNING
                } else {
                    theme::DANGER
                },
            );
        }
        for row in tiles.chunks(columns) {
            ui.horizontal(|ui| {
                for (label, value, color) in row {
                    ui.allocate_ui_with_layout(
                        egui::vec2(156.0, theme::CONTROL_HEIGHT),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            ui.set_width(156.0);
                            theme::metric(ui, label, value, *color);
                        },
                    );
                }
            });
        }
    }
}
/// Narrowest workspace with all four metric chips on one row.
const METRICS_ONE_ROW_WIDTH: f32 = 680.0;
/// Picture of the About dialog, when assets/about.png or .jpg existed at build time.
#[cfg(about_image)]
const ABOUT_IMAGE: Option<&[u8]> = Some(include_bytes!(env!("OXIDE_ABOUT_IMAGE")));
#[cfg(not(about_image))]
const ABOUT_IMAGE: Option<&[u8]> = None;
/// Width of the settings column.
const SETTINGS_WIDTH: f32 = 560.0;
/// Saved normal window size in points and maximized state.
fn saved_window(saved: &serde_json::Map<String, serde_json::Value>) -> (Option<[f32; 2]>, bool) {
    let read = settings::Reader(saved);
    let (mut width, mut height, mut maximized) = (0.0, 0.0, false);
    read.number("window_width", 800.0..=10000.0, &mut width);
    read.number("window_height", 600.0..=10000.0, &mut height);
    read.flag("window_maximized", &mut maximized);
    let size = (width > 0.0 && height > 0.0).then_some([width as f32, height as f32]);
    (size, maximized)
}
fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|s| s == "--stream-rf") {
        if let Err(e) = vhs_opencl::network::serve(&args[1..]) {
            eprintln!("RF sender: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "--export-mkv") {
        if let Err(e) = vhs_opencl::export::run(&args[1..]) {
            eprintln!("MKV export: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "--decode-native") {
        match vhs_opencl::native::decode(&args[1..]) {
            Ok(()) => return Ok(()),
            Err(e) => {
                eprintln!("Decoding: {e}");
                std::process::exit(1);
            }
        }
    }
    if args.first().is_some_and(|s| s == "--diagnose") {
        println!("Oxide RF · Native Rust · no Python");
        println!("OpenCL GPU devices: {:?}", vhs_opencl::devices());
        for device in vhs_opencl::cxadc::devices() {
            println!("cxadc {}: {}", device.path.display(), device.summary());
        }
        return Ok(());
    }
    // Window geometry is needed before the window exists.
    let saved = settings::load();
    let (size, maximized) = saved_window(&saved);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(size.unwrap_or([1600.0, 900.0]))
            .with_app_id("oxide-rf")
            .with_title(concat!("Oxide RF ", env!("CARGO_PKG_VERSION")))
            .with_min_inner_size([1180.0, 720.0])
            .with_maximized(maximized)
            // eframe derives the monitor size from Wayland's integer output
            // scale (2 at 125%), which would cap a 1440-pixel-high screen at
            // 720 points. Saved sizes are already range checked.
            .with_clamp_size_to_monitor_size(false),
        ..Default::default()
    };
    eframe::run_native(
        "Oxide RF",
        options,
        Box::new(move |cc| {
            theme::apply(&cc.egui_ctx);
            let mut app = App::default();
            app.apply_settings(&saved);
            app.saved_settings = app.settings_json();
            #[cfg(feature = "ui-screenshot")]
            {
                app.preview_colour = std::env::var_os("VHS_GUI_COLOUR").is_some();
                app.full_raster = std::env::var_os("VHS_GUI_FULL_RASTER").is_some();
                app.vbi_enabled = std::env::var_os("VHS_GUI_VBI_FIXTURE").is_some();
                app.burn_teletext = app.vbi_enabled;
                app.show_settings = std::env::var_os("VHS_GUI_SETTINGS_DIALOG").is_some();
                app.dropout_correct = std::env::var_os("VHS_GUI_DROPOUT_FIXTURE").is_some();
            }
            #[cfg(feature = "ui-screenshot")]
            if let Ok(path) = std::env::var("VHS_GUI_PREVIEW_FIXTURE") {
                app.preview_path = Some(path.into());
                app.refresh_preview(&cc.egui_ctx);
                app.preview_path = None;
            }
            #[cfg(feature = "ui-screenshot")]
            if let Ok(path) = std::env::var("VHS_GUI_LIVE_FIXTURE") {
                app.live = true;
                app.live_input = path;
                app.sample_format = "u8".into();
            }
            #[cfg(feature = "ui-screenshot")]
            if std::env::var_os("VHS_GUI_SERVER_FIXTURE").is_some() {
                app.server = true;
                app.live = true;
                app.preview_only = true;
                app.server_audio = std::env::var_os("VHS_GUI_AUDIO_FIXTURE").is_some();
            }
            #[cfg(feature = "ui-screenshot")]
            if let (Ok(input), Ok(output)) = (
                std::env::var("VHS_GUI_EXPORT_RF_FIXTURE"),
                std::env::var("VHS_GUI_EXPORT_OUTPUT_FIXTURE"),
            ) {
                app.choose_input(input.into());
                app.output = output;
                app.frames = 3;
                app.export_after_decode = true;
                app.direct_mkv = std::env::var_os("VHS_GUI_DIRECT_MKV_FIXTURE").is_some();
                if let Err(e) = app.launch() {
                    app.status = e;
                    app.started = Some(Instant::now());
                }
            }
            #[cfg(feature = "ui-screenshot")]
            if let Ok(input) = std::env::var("VHS_GUI_NETWORK_FIXTURE") {
                app.network = true;
                app.network_input = input;
                app.preview_only = true;
                app.sample_format = "auto".into();
                if let Ok(output) = std::env::var("VHS_GUI_NETWORK_OUTPUT") {
                    app.preview_only = false;
                    app.output = output;
                    app.export_after_decode = std::env::var_os("VHS_GUI_NETWORK_MKV").is_some();
                    app.direct_mkv = std::env::var_os("VHS_GUI_NETWORK_DIRECT_MKV").is_some();
                    app.dropout_correct = std::env::var_os("VHS_GUI_DROPOUT_FIXTURE").is_some();
                }
                if let Err(e) = app.launch() {
                    app.status = e;
                    app.started = Some(Instant::now());
                }
            }
            #[cfg(feature = "ui-screenshot")]
            if let Ok(input) = std::env::var("VHS_GUI_MONITOR_FIXTURE") {
                app.live = vhs_opencl::cxadc::is_stream(std::path::Path::new(&input));
                app.input = input.clone();
                app.live_input = input;
                app.sample_format = "u8".into();
                app.preview_only = true;
                app.output = String::new();
                if let Err(e) = app.launch() {
                    app.status = e;
                    app.started = Some(Instant::now());
                }
            }
            // Keep initialization identical in normal and screenshot builds.
            app.preview_checked = Instant::now();
            Ok(Box::new(app))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Drives the header editor with a mouse drag from `from` to `to`.
    fn drag_header(app: &mut App, area: egui::Rect, from: egui::Pos2, to: egui::Pos2) {
        let ctx = egui::Context::default();
        let press = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };
        let steps = [
            vec![egui::Event::PointerMoved(from)],
            vec![press(from, true)],
            vec![egui::Event::PointerMoved(from + (to - from) * 0.5)],
            vec![egui::Event::PointerMoved(to)],
            vec![press(to, false)],
            vec![],
        ];
        for events in steps {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1000.0, 800.0),
                )),
                events,
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    app.header_editor(ui, area, preview::WIDTH);
                });
            });
        }
    }
    #[test]
    fn header_box_moves_and_resizes_from_its_corners() {
        let area = egui::Rect::from_min_size(egui::pos2(100.0, 100.0), egui::vec2(800.0, 600.0));
        let screen = |app: &App| {
            let [x, y, w, h] =
                preview::HeaderOverlay::placement(preview::WIDTH, app.header_placement);
            egui::Rect::from_min_size(
                area.min + egui::vec2(x * 800.0, y * 600.0),
                egui::vec2(w * 800.0, h * 600.0),
            )
        };
        let mut app = App::default();
        let start = screen(&app);
        // Dragging the body moves the box by the pointer distance.
        drag_header(
            &mut app,
            area,
            start.center(),
            start.center() + egui::vec2(-40.0, 120.0),
        );
        let moved = screen(&app);
        assert!((moved.min - (start.min + egui::vec2(-40.0, 120.0))).length() < 0.5);
        assert_eq!(app.header_placement.size, 1.0);
        // Dragging the bottom-right corner grows the box; the top-left stays.
        let target = moved.right_bottom() + egui::vec2(50.0, 0.0);
        drag_header(&mut app, area, moved.right_bottom(), target);
        let grown = screen(&app);
        assert!((grown.min - moved.min).length() < 0.5);
        let expected = (target.x - moved.left()) / moved.width();
        assert!((app.header_placement.size - expected).abs() < 1e-3);
        assert!((grown.width() / grown.height() - moved.width() / moved.height()).abs() < 1e-3);
        // Dragging the top-left corner shrinks it towards the fixed bottom-right.
        let target = grown.left_top() + egui::vec2(grown.width() * 0.5, 0.0);
        drag_header(&mut app, area, grown.left_top(), target);
        let shrunk = screen(&app);
        assert!((shrunk.right_bottom() - grown.right_bottom()).length() < 0.5);
        assert!((shrunk.width() - grown.width() * 0.5).abs() < 0.5);
        // Neither moving nor resizing can take the header past the picture.
        drag_header(
            &mut app,
            area,
            shrunk.center(),
            shrunk.center() + egui::vec2(900.0, 900.0),
        );
        let edge = screen(&app);
        assert!((edge.right_bottom() - area.right_bottom()).length() < 0.5);
        drag_header(&mut app, area, edge.left_top(), egui::pos2(-500.0, -500.0));
        let largest = screen(&app);
        assert!(area.expand(0.5).contains_rect(largest));
        assert!((largest.left() - area.left()).abs() < 0.5);
        assert!((largest.right_bottom() - area.right_bottom()).length() < 0.5);
    }
    #[test]
    fn settings_round_trip_and_reject_invalid_values() {
        let mut app = App::default();
        (app.server, app.live, app.network) = (false, false, true);
        app.network_input = "tcp://10.0.0.2:5555".into();
        app.output = "/home/user/Videos/tape".into();
        app.preview_only = false;
        app.direct_mkv = true;
        app.dropout_correct = true;
        app.dropout_threshold = Some(0.4);
        app.automatic_black = true;
        app.export_profile = "viewing".into();
        app.audio_offset_ms = -120.0;
        app.frames = 250;
        let saved = app.settings_json();
        let mut restored = App::default();
        restored.apply_settings(saved.as_object().unwrap());
        assert_eq!(restored.settings_json(), saved);

        let mut invalid = saved.as_object().unwrap().clone();
        invalid.insert("frequency_mhz".into(), serde_json::json!(500.0));
        invalid.insert("mkv_profile".into(), serde_json::json!("bogus"));
        invalid.insert("dropout_threshold".into(), serde_json::json!(9));
        invalid.insert("frames".into(), serde_json::json!(-1));
        invalid.insert("gpu".into(), serde_json::json!("yes"));
        let mut checked = App::default();
        checked.apply_settings(&invalid);
        assert_eq!(checked.frequency, 315.0 / 11.0);
        assert_eq!(checked.export_profile, "archive");
        assert_eq!(checked.dropout_threshold, None);
        assert_eq!(checked.frames, 0);
        assert!(checked.gpu);
        assert!(checked.network && !checked.live && !checked.server);
    }
}
