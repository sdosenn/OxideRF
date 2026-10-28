pub mod audio;
mod burn;
pub mod channel;
pub mod cxadc;
mod diagnostics;
pub mod dropout;
pub mod export;
pub mod network;
pub mod preview;
pub mod vbi;
use ocl::{flags, Buffer, Device, Kernel, Platform, ProQue};
mod gpu;
pub mod native;
mod resample;
use std::{
    collections::HashMap,
    ffi::CString,
    os::raw::c_char,
    sync::{Mutex, OnceLock},
};

struct Plan {
    input: Buffer<f32>,
    output: Buffer<f32>,
    kernels: [Vec<Kernel>; 2],
}
struct Engine {
    pq: ProQue,
    plans: HashMap<usize, Plan>,
}
impl Engine {
    fn new() -> Result<Self, String> {
        let index: usize = std::env::var("VHS_GPU_DEVICE")
            .unwrap_or_default()
            .parse()
            .unwrap_or(0);
        let mut devices = Vec::new();
        for p in platforms() {
            for d in Device::list(p, Some(flags::DEVICE_TYPE_GPU)).unwrap_or_default() {
                devices.push((p, d));
            }
        }
        let (p, d) = devices.get(index).ok_or("No OpenCL GPU device available")?;
        let pq = ProQue::builder()
            .platform(*p)
            .device(*d)
            .src(format!(
                "{}\n{}\n{}",
                include_str!("../kernels/fft.cl"),
                include_str!("../kernels/decode.cl"),
                include_str!("../kernels/export.cl")
            ))
            .dims(1)
            .build()
            .map_err(|e| e.to_string())?;
        eprintln!("OpenCL GPU: {} (FP32 FFT)", d.name().unwrap_or_default());
        Ok(Self {
            pq,
            plans: HashMap::new(),
        })
    }
    fn fft(&mut self, input: &[f32], output: &mut [f32], inverse: bool) -> ocl::Result<()> {
        let n = input.len() / 2;
        if !self.plans.contains_key(&n) {
            let buf = || {
                Buffer::<f32>::builder()
                    .queue(self.pq.queue().clone())
                    .len(n * 2)
                    .build()
            };
            let mut tw = Vec::with_capacity(n);
            for k in 0..n / 2 {
                let a = -std::f64::consts::TAU * k as f64 / n as f64;
                tw.extend([a.cos() as f32, a.sin() as f32]);
            }
            let twiddle = Buffer::builder()
                .queue(self.pq.queue().clone())
                .len(n)
                .copy_host_slice(&tw)
                .build()?;
            let input = buf()?;
            let output = buf()?;
            let mut kernels = [Vec::new(), Vec::new()];
            for (inverse, direction_kernels) in kernels.iter_mut().enumerate() {
                let first_stage = if n >= 512 {
                    direction_kernels.push(
                        self.pq
                            .kernel_builder("local_begin")
                            .global_work_size(n / 2)
                            .local_work_size(256)
                            .arg(&input)
                            .arg(&output)
                            .arg(&twiddle)
                            .arg(n.trailing_zeros())
                            .arg(n as u32)
                            .arg(inverse as i32)
                            .build()?,
                    );
                    10
                } else {
                    direction_kernels.push(
                        self.pq
                            .kernel_builder("reorder")
                            .global_work_size(n)
                            .arg(&input)
                            .arg(&output)
                            .arg(n.trailing_zeros())
                            .arg(n as u32)
                            .build()?,
                    );
                    1
                };
                if n > 512 && n <= 65536 {
                    direction_kernels.push(
                        self.pq
                            .kernel_builder("local_end")
                            .global_work_size(n / 2)
                            .local_work_size(n / 1024)
                            .arg(&output)
                            .arg(&twiddle)
                            .arg(n as u32)
                            .arg(inverse as i32)
                            .build()?,
                    );
                    continue;
                }
                for b in first_stage..=n.trailing_zeros() {
                    direction_kernels.push(
                        self.pq
                            .kernel_builder("stage")
                            .global_work_size(n / 2)
                            .arg(&output)
                            .arg(&twiddle)
                            .arg(n as u32)
                            .arg(1u32 << b)
                            .arg(inverse as i32)
                            .build()?,
                    );
                }
            }
            self.plans.insert(
                n,
                Plan {
                    input,
                    output,
                    kernels,
                },
            );
        }
        let p = &self.plans[&n];
        p.input.write(input).enq()?;
        // All kernels operate on validated, equally sized buffers, on one in-order queue.
        for kernel in &p.kernels[inverse as usize] {
            unsafe {
                kernel.enq()?;
            }
        }
        p.output.read(&mut *output).enq()?;
        if inverse {
            for x in output {
                *x /= n as f32;
            }
        }
        Ok(())
    }
}
static ENGINE: OnceLock<Mutex<Option<Engine>>> = OnceLock::new();
static ERROR: Mutex<String> = Mutex::new(String::new());

/// C ABI: interleaved real/imaginary FP32 arrays of 2*n elements. Buffers must
/// be valid, non-overlapping, and writable for the duration of this call.
///
/// # Safety
/// `src` and `dst` must reference non-overlapping allocations of at least 2*n
/// aligned f32 elements. `dst` must be writable; both must outlive the call.
#[no_mangle]
pub unsafe extern "C" fn vhs_fft(src: *const f32, dst: *mut f32, n: usize, inverse: i32) -> i32 {
    let result = std::panic::catch_unwind(|| -> Result<(), String> {
        if src.is_null() || dst.is_null() || n < 2 || !n.is_power_of_two() || n > (1 << 22) {
            return Err("FFT requires a power of two (2..4194304)".into());
        }
        let mut guard = ENGINE
            .get_or_init(|| Mutex::new(None))
            .lock()
            .map_err(|e| e.to_string())?;
        if guard.is_none() {
            *guard = Some(Engine::new()?);
        }
        guard
            .as_mut()
            .unwrap()
            .fft(
                std::slice::from_raw_parts(src, n * 2),
                std::slice::from_raw_parts_mut(dst, n * 2),
                inverse != 0,
            )
            .map_err(|e| e.to_string())
    });
    match result {
        Ok(Ok(())) => 0,
        r => {
            if let Ok(mut error) = ERROR.lock() {
                *error = match r {
                    Ok(Err(e)) => e,
                    _ => "OpenCL panic".into(),
                };
            }
            -1
        }
    }
}
#[no_mangle]
pub extern "C" fn vhs_error() -> *mut c_char {
    CString::new(ERROR.lock().unwrap().replace('\0', " "))
        .unwrap()
        .into_raw()
}
#[no_mangle]
/// Release a diagnostic string returned by `vhs_error`.
///
/// # Safety
/// `p` must be null or an unfreed pointer returned by `vhs_error`.
pub unsafe extern "C" fn vhs_free_error(p: *mut c_char) {
    if !p.is_null() {
        drop(CString::from_raw(p));
    }
}

pub fn devices() -> Vec<String> {
    let mut result = Vec::new();
    for p in platforms() {
        if let Ok(ds) = Device::list(p, Some(flags::DEVICE_TYPE_GPU)) {
            result.extend(ds.into_iter().filter_map(|d| d.name().ok()));
        }
    }
    result
}

fn platforms() -> Vec<Platform> {
    ocl::core::get_platform_ids()
        .unwrap_or_default()
        .into_iter()
        .map(Platform::new)
        .collect()
}

// Avoid requiring Arch's new GLIBC_2.43 float-math symbol versions on Zorin 18.
// The double precision equivalents have been available since GLIBC_2.2.5.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[no_mangle]
pub extern "C" fn __wrap_atan2f(y: f32, x: f32) -> f32 {
    (y as f64).atan2(x as f64) as f32
}
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[no_mangle]
pub extern "C" fn __wrap_acosf(x: f32) -> f32 {
    (x as f64).acos() as f32
}
