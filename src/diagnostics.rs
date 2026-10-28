//! Periodic live-capture diagnostics. Lines are printed and appended to
//! `<output>.live-stats.log`, so an RF overflow can still be explained after the
//! GUI activity log has scrolled past it.
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const PERIOD: Duration = Duration::from_secs(10);

/// Cumulative decoder counters at one point in time.
#[derive(Clone, Copy, Default)]
pub(crate) struct Sample {
    pub fields: usize,
    pub skipped: usize,
    pub estimated: usize,
    pub dropout_fields: usize,
    pub demod: Duration,
    pub field: Duration,
    pub input_wait: Duration,
    pub received: u64,
    pub queued: usize,
    pub capacity: usize,
    pub mkv_queue: usize,
    pub mkv_send_ns: u64,
    pub mkv_render_ns: u64,
    pub mkv_write_ns: u64,
}

struct Gpu {
    frequency: PathBuf,
    throttle: Option<PathBuf>,
    temperatures: Vec<PathBuf>,
}

pub(crate) struct LiveStats {
    file: Option<File>,
    started: Instant,
    last_at: Instant,
    last: Sample,
    bytes_per_second: f64,
    tick: f64,
    self_cpu: Option<u64>,
    encoder_cpu: Option<u64>,
    encoder_pid: u32,
    gpu: Option<Gpu>,
    worst_headroom: f64,
    swap: Option<(u64, u64)>,
    major_faults: Option<u64>,
    processes: HashMap<u32, (String, u64)>,
    host: Host,
}

/// Cumulative kernel counters for scheduler and CPU diagnostics.
#[derive(Clone, Default)]
struct Host {
    /// Per process: thread count, total run-queue wait ns.
    decoder: (usize, u64),
    encoder: (usize, u64),
    forks: u64,
    switches: u64,
    /// user, nice, system, idle, iowait, irq, softirq, steal jiffies.
    cpu: [u64; 8],
}
impl Host {
    fn read(encoder_pid: u32) -> Self {
        let stat = fs::read_to_string("/proc/stat").unwrap_or_default();
        let counter = |key: &str| {
            stat.lines()
                .find_map(|l| l.strip_prefix(key)?.trim().parse::<u64>().ok())
                .unwrap_or(0)
        };
        let mut cpu = [0; 8];
        if let Some(line) = stat.lines().find(|l| l.starts_with("cpu ")) {
            for (slot, value) in cpu.iter_mut().zip(line.split_whitespace().skip(1)) {
                *slot = value.parse().unwrap_or(0);
            }
        }
        Self {
            decoder: threads("self"),
            encoder: if encoder_pid == 0 {
                (0, 0)
            } else {
                threads(&encoder_pid.to_string())
            },
            forks: counter("processes "),
            switches: counter("ctxt "),
            cpu,
        }
    }
}
/// Thread count and summed run-queue wait of all threads of a process.
fn threads(process: &str) -> (usize, u64) {
    let mut count = 0;
    let mut wait = 0;
    for task in fs::read_dir(format!("/proc/{process}/task"))
        .into_iter()
        .flatten()
        .flatten()
    {
        count += 1;
        wait += fs::read_to_string(task.path().join("schedstat"))
            .ok()
            .and_then(|v| v.split_whitespace().nth(1)?.parse::<u64>().ok())
            .unwrap_or(0);
    }
    (count, wait)
}
/// Average current CPU clock in MHz and the hottest CPU sensor in °C.
fn cpu_clock_and_temperature() -> (Option<f64>, Option<f64>) {
    let clocks: Vec<f64> = fs::read_dir("/sys/devices/system/cpu")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|cpu| {
            fs::read_to_string(cpu.path().join("cpufreq/scaling_cur_freq"))
                .ok()?
                .trim()
                .parse::<f64>()
                .ok()
        })
        .collect();
    let clock =
        (!clocks.is_empty()).then(|| clocks.iter().sum::<f64>() / clocks.len() as f64 / 1000.0);
    let temperature = fs::read_dir("/sys/class/hwmon")
        .into_iter()
        .flatten()
        .flatten()
        .filter(|hwmon| {
            fs::read_to_string(hwmon.path().join("name"))
                .is_ok_and(|name| ["k10temp", "coretemp", "zenpower"].contains(&name.trim()))
        })
        .filter_map(|hwmon| {
            fs::read_to_string(hwmon.path().join("temp1_input"))
                .ok()?
                .trim()
                .parse::<f64>()
                .ok()
        })
        .fold(None, |max: Option<f64>, v| {
            Some(max.map_or(v, |m| m.max(v)))
        })
        .map(|v| v / 1000.0);
    (clock, temperature)
}
impl LiveStats {
    pub(crate) fn new(
        base: &Path,
        bytes_per_second: f64,
        encoder_pid: u32,
        settings: &str,
    ) -> Self {
        let file = (!base.as_os_str().is_empty())
            .then(|| {
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(format!("{}.live-stats.log", base.display()))
            })
            .and_then(|file| file.map_err(|e| eprintln!("Live stats file: {e}")).ok());
        // Clock ticks of /proc CPU counters; those counters exist only on Linux.
        #[cfg(unix)]
        let tick = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as f64;
        #[cfg(not(unix))]
        let tick = 100.0;
        let mut stats = Self {
            file,
            started: Instant::now(),
            last_at: Instant::now(),
            last: Sample::default(),
            bytes_per_second,
            tick,
            self_cpu: cpu_ticks("self"),
            encoder_cpu: None,
            encoder_pid,
            gpu: gpu_paths(),
            worst_headroom: f64::INFINITY,
            swap: swap_pages(),
            major_faults: major_faults(),
            processes: processes(),
            host: Host::read(encoder_pid),
        };
        stats.encoder_cpu = stats.encoder_ticks();
        stats.emit(&format!(
            "LIVE_STATS_START version={} unix_time={} {settings}",
            env!("CARGO_PKG_VERSION"),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs())
        ));
        stats
    }
    fn encoder_ticks(&self) -> Option<u64> {
        (self.encoder_pid != 0)
            .then(|| cpu_ticks(&self.encoder_pid.to_string()))
            .flatten()
    }
    fn emit(&mut self, line: &str) {
        println!("{line}");
        if let Some(file) = &mut self.file {
            if writeln!(file, "{line}").is_err() {
                self.file = None;
            }
        }
    }
    /// Report once per period; cheap to call for every RF batch.
    pub(crate) fn update(&mut self, now: Sample) {
        if self.last_at.elapsed() >= PERIOD {
            self.report(now);
        }
    }
    fn report(&mut self, now: Sample) {
        let wall = self.last_at.elapsed().as_secs_f64().max(1e-6);
        let last = self.last;
        let fields = now.fields.saturating_sub(last.fields);
        let per_field = |ns: f64| {
            if fields == 0 {
                0.0
            } else {
                ns / 1e6 / fields as f64
            }
        };
        let work = (now.demod + now.field)
            .saturating_sub(last.demod + last.field)
            .as_secs_f64()
            .max(1e-6);
        let headroom = fields as f64 * 0.02 / work;
        if fields > 0 {
            self.worst_headroom = self.worst_headroom.min(headroom);
        }
        let self_cpu = cpu_ticks("self");
        let encoder_cpu = self.encoder_ticks();
        let percent = |now: Option<u64>, last: Option<u64>| match (now, last) {
            (Some(now), Some(last)) => {
                format!("{:.0}", (now - last) as f64 / self.tick / wall * 100.0)
            }
            _ => "?".into(),
        };
        let gpu = self
            .gpu
            .as_ref()
            .map_or("gpu_mhz=? gpu_temp_c=? gpu_throttle=?".into(), |gpu| {
                let read = |path: &Path| fs::read_to_string(path).ok().map(|v| v.trim().to_owned());
                let temperature = gpu
                    .temperatures
                    .iter()
                    .filter_map(|path| read(path)?.parse::<f64>().ok())
                    .fold(None, |max: Option<f64>, v| {
                        Some(max.map_or(v, |m| m.max(v)))
                    });
                format!(
                    "gpu_mhz={} gpu_temp_c={} gpu_throttle={}",
                    read(&gpu.frequency).unwrap_or_else(|| "?".into()),
                    temperature.map_or("?".into(), |t| format!("{:.0}", t / 1000.0)),
                    gpu.throttle
                        .as_deref()
                        .and_then(read)
                        .unwrap_or_else(|| "?".into())
                        .replace(' ', ",")
                )
            });
        let load = fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|v| v.split_whitespace().next().map(str::to_owned))
            .unwrap_or_else(|| "?".into());
        let memory = |key: &str| {
            fs::read_to_string("/proc/meminfo").ok().and_then(|v| {
                v.lines().find_map(|l| {
                    l.strip_prefix(key)?
                        .split_whitespace()
                        .next()?
                        .parse::<u64>()
                        .ok()
                })
            })
        };
        let stat = |key: &str| {
            fs::read_to_string("/proc/stat").ok().and_then(|v| {
                v.lines()
                    .find_map(|l| l.strip_prefix(key)?.trim().parse::<u64>().ok())
            })
        };
        let swap = swap_pages();
        let swap_mib_s = |pick: fn((u64, u64)) -> u64| match (swap, self.swap) {
            (Some(now), Some(last)) => {
                format!(
                    "{:.1}",
                    pick(now).saturating_sub(pick(last)) as f64 * 4096.0 / 1048576.0 / wall
                )
            }
            _ => "?".into(),
        };
        let faults = major_faults();
        // Other processes using the most CPU since the previous line, to expose
        // competing work outside the decoder.
        let current = processes();
        let own = [std::process::id(), self.encoder_pid];
        let mut busy: Vec<_> = current
            .iter()
            .filter(|(pid, _)| !own.contains(pid))
            .filter_map(|(pid, (name, ticks))| {
                let (_, previous) = self.processes.get(pid)?;
                let pct = ticks.saturating_sub(*previous) as f64 / self.tick / wall * 100.0;
                (pct >= 5.0).then(|| (pct, format!("{name}:{pid}")))
            })
            .collect();
        busy.sort_by(|a, b| b.0.total_cmp(&a.0));
        let top = busy
            .iter()
            .take(3)
            .map(|(pct, name)| format!("{name}={pct:.0}%"))
            .collect::<Vec<_>>()
            .join(",");
        let system = format!(
            "mem_avail_mib={} swap_in_mib_s={} swap_out_mib_s={} decoder_major_faults={} procs_running={} procs_blocked={} top_other_cpu={}",
            memory("MemAvailable:").map_or("?".into(), |kib| (kib / 1024).to_string()),
            swap_mib_s(|s| s.0),
            swap_mib_s(|s| s.1),
            match (faults, self.major_faults) {
                (Some(now), Some(last)) => now.saturating_sub(last).to_string(),
                _ => "?".into(),
            },
            stat("procs_running").map_or("?".into(), |v| v.to_string()),
            stat("procs_blocked").map_or("?".into(), |v| v.to_string()),
            if top.is_empty() { "none".into() } else { top },
        );
        let host = Host::read(self.encoder_pid);
        let previous = &self.host;
        let jiffies: Vec<u64> = host
            .cpu
            .iter()
            .zip(previous.cpu)
            .map(|(now, last)| now.saturating_sub(last))
            .collect();
        let total = jiffies.iter().sum::<u64>().max(1) as f64;
        let share = |index: usize| jiffies[index] as f64 * 100.0 / total;
        let wait_ms_s = |now: u64, last: u64| now.saturating_sub(last) as f64 / 1e6 / wall;
        let (clock, temperature) = cpu_clock_and_temperature();
        let scheduler = format!(
            "decoder_threads={} decoder_runq_wait_ms_s={:.0} ffmpeg_threads={} ffmpeg_runq_wait_ms_s={:.0} \
             forks_s={:.0} ctxt_s={:.0} cpu_user_pct={:.0} cpu_sys_pct={:.0} cpu_iowait_pct={:.0} cpu_irq_pct={:.0} \
             cpu_softirq_pct={:.0} cpu_idle_pct={:.0} cpu_mhz={} cpu_temp_c={}",
            host.decoder.0,
            wait_ms_s(host.decoder.1, previous.decoder.1),
            host.encoder.0,
            wait_ms_s(host.encoder.1, previous.encoder.1),
            host.forks.saturating_sub(previous.forks) as f64 / wall,
            host.switches.saturating_sub(previous.switches) as f64 / wall,
            share(0) + share(1),
            share(2),
            share(4),
            share(5),
            share(6),
            share(3),
            clock.map_or("?".into(), |v| format!("{v:.0}")),
            temperature.map_or("?".into(), |v| format!("{v:.0}")),
        );
        let line = format!(
            "LIVE_STATS t_s={:.0} headroom={headroom:.2} rf_backlog_s={:.2} rf_buffer_pct={:.1} rf_in_mib_s={:.2} \
             fields={fields} skipped={} estimated={} dropout_fields={} \
             demod_ms={:.2} tbc_ms={:.2} mkv_send_ms={:.2} mkv_render_ms={:.2} mkv_write_ms={:.2} mkv_queue={} \
             input_wait_pct={:.0} cpu_decoder_pct={} cpu_ffmpeg_pct={} load1={load} {gpu} {system} {scheduler}",
            self.started.elapsed().as_secs_f64(),
            now.queued as f64 / self.bytes_per_second.max(1.0),
            now.queued as f64 * 100.0 / now.capacity.max(1) as f64,
            now.received.saturating_sub(last.received) as f64 / 1048576.0 / wall,
            now.skipped - last.skipped,
            now.estimated - last.estimated,
            now.dropout_fields - last.dropout_fields,
            per_field(now.demod.saturating_sub(last.demod).as_nanos() as f64),
            per_field(now.field.saturating_sub(last.field).as_nanos() as f64),
            per_field((now.mkv_send_ns - last.mkv_send_ns) as f64),
            per_field((now.mkv_render_ns - last.mkv_render_ns) as f64),
            per_field((now.mkv_write_ns - last.mkv_write_ns) as f64),
            now.mkv_queue,
            now.input_wait.saturating_sub(last.input_wait).as_secs_f64() * 100.0 / wall,
            percent(self_cpu, self.self_cpu),
            percent(encoder_cpu, self.encoder_cpu),
        );
        self.emit(&line);
        self.last = now;
        self.last_at = Instant::now();
        self.self_cpu = self_cpu;
        self.encoder_cpu = encoder_cpu;
        self.swap = swap;
        self.major_faults = faults;
        self.processes = current;
        self.host = host;
    }
    /// Write the final partial period and why acquisition ended.
    pub(crate) fn finish(&mut self, now: Sample, error: Option<&str>) {
        self.report(now);
        let line = format!(
            "LIVE_STATS_END elapsed_s={:.1} fields={} video_s={:.2} rf_received_s={:.2} rf_backlog_s={:.2} \
             worst_headroom={} skipped={} estimated={} dropout_fields={} result={}",
            self.started.elapsed().as_secs_f64(),
            now.fields,
            now.fields as f64 * 0.02,
            now.received as f64 / self.bytes_per_second.max(1.0),
            now.queued as f64 / self.bytes_per_second.max(1.0),
            if self.worst_headroom.is_finite() {
                format!("{:.2}", self.worst_headroom)
            } else {
                "?".into()
            },
            now.skipped,
            now.estimated,
            now.dropout_fields,
            error.map_or("ok".into(), |e| format!("error: {e}")),
        );
        self.emit(&line);
    }
}

/// Fields of `/proc/<process>/stat` after the command name, which may contain
/// spaces, plus that name.
fn process_stat(process: &str) -> Option<(String, Vec<u64>)> {
    let stat = fs::read_to_string(format!("/proc/{process}/stat")).ok()?;
    let (head, tail) = stat.rsplit_once(')')?;
    let name = head.split_once('(')?.1.replace([' ', ',', '='], "_");
    // The state letter is the first field; keep numeric positions stable.
    let fields = tail
        .split_whitespace()
        .skip(1)
        .map(|v| v.parse().unwrap_or(0))
        .collect();
    Some((name, fields))
}
/// User plus system CPU ticks of a process (`self` or a pid).
fn cpu_ticks(process: &str) -> Option<u64> {
    let (_, fields) = process_stat(process)?;
    Some(fields.get(10)? + fields.get(11)?)
}
fn major_faults() -> Option<u64> {
    process_stat("self")?.1.get(8).copied()
}
fn processes() -> HashMap<u32, (String, u64)> {
    fs::read_dir("/proc")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let pid = entry.file_name().to_str()?.parse::<u32>().ok()?;
            let (name, fields) = process_stat(&pid.to_string())?;
            Some((pid, (name, fields.get(10)? + fields.get(11)?)))
        })
        .collect()
}
/// Pages swapped in and out since boot.
fn swap_pages() -> Option<(u64, u64)> {
    let vmstat = fs::read_to_string("/proc/vmstat").ok()?;
    let value = |key: &str| {
        vmstat
            .lines()
            .find_map(|l| l.strip_prefix(key)?.trim().parse::<u64>().ok())
    };
    Some((value("pswpin ")?, value("pswpout ")?))
}

/// Frequency, throttle and temperature files of the first Intel xe/i915 GPU.
fn gpu_paths() -> Option<Gpu> {
    let mut cards: Vec<_> = fs::read_dir("/sys/class/drm")
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_prefix("card"))
                .is_some_and(|n| n.bytes().all(|c| c.is_ascii_digit()))
        })
        .collect();
    cards.sort();
    cards.into_iter().find_map(|card| {
        let xe = card.join("device/tile0/gt0/freq0");
        let (frequency, throttle) = if xe.join("act_freq").exists() {
            (xe.join("act_freq"), Some(xe.join("throttle/reasons")))
        } else if card.join("gt_act_freq_mhz").exists() {
            (card.join("gt_act_freq_mhz"), None)
        } else {
            return None;
        };
        let temperatures = fs::read_dir(card.join("device/hwmon"))
            .into_iter()
            .flatten()
            .flatten()
            .flat_map(|hwmon| fs::read_dir(hwmon.path()).into_iter().flatten().flatten())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("temp") && n.ends_with("_input"))
            })
            .collect();
        Some(Gpu {
            frequency,
            throttle: throttle.filter(|path| path.exists()),
            temperatures,
        })
    })
}
