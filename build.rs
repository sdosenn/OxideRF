use std::{env, path::PathBuf};

fn main() {
    // An optional picture for the About dialog: assets/about.png or .jpg.
    println!("cargo:rustc-check-cfg=cfg(about_image)");
    println!("cargo:rerun-if-changed=assets");
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    if let Some(image) = ["about.png", "about.jpg"]
        .iter()
        .map(|name| root.join("assets").join(name))
        .find(|path| path.is_file())
    {
        println!("cargo:rustc-cfg=about_image");
        println!("cargo:rustc-env=OXIDE_ABOUT_IMAGE={}", image.display());
    }
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os == "linux" && env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("gnu") {
        println!("cargo:rustc-link-arg=-Wl,--wrap=atan2f");
        println!("cargo:rustc-link-arg=-Wl,--wrap=acosf");
    }
    if os == "windows" {
        // The .exe icon shown by Explorer; the window icon is set at runtime.
        #[cfg(windows)]
        {
            let mut resource = winresource::WindowsResource::new();
            resource.set_icon("assets/icon.ico");
            if let Err(err) = resource.compile_for("oxide-rf") {
                println!("cargo:warning=Windows icon resource not embedded: {err}");
            }
        }
        // GPU drivers install OpenCL.dll but not the OpenCL.lib import library
        // needed to link. Look in OPENCL_LIB_DIR, then in a vcpkg installation
        // (`vcpkg install opencl:x64-windows`).
        println!("cargo:rerun-if-env-changed=OPENCL_LIB_DIR");
        println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
        println!("cargo:rerun-if-env-changed=VCPKG_INSTALLATION_ROOT");
        let triplet = if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64") {
            "arm64-windows"
        } else {
            "x64-windows"
        };
        let candidates = env::var_os("OPENCL_LIB_DIR")
            .map(PathBuf::from)
            .into_iter()
            .chain(
                ["VCPKG_ROOT", "VCPKG_INSTALLATION_ROOT"]
                    .into_iter()
                    .filter_map(env::var_os)
                    .map(|root| {
                        PathBuf::from(root)
                            .join("installed")
                            .join(triplet)
                            .join("lib")
                    }),
            );
        match candidates.into_iter().find(|dir| dir.join("OpenCL.lib").is_file()) {
            Some(dir) => println!("cargo:rustc-link-search=native={}", dir.display()),
            None => println!(
                "cargo:warning=OpenCL.lib not found; set OPENCL_LIB_DIR or run `vcpkg install opencl:{triplet}`"
            ),
        }
    }
}
