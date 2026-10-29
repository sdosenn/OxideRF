#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Keep the downloaded tools in the ignored build directory so they survive reboots.
tool="${APPIMAGETOOL:-$PWD/build/tools/appimagetool-x86_64.AppImage}"
runtime="${APPIMAGE_RUNTIME:-$PWD/build/tools/runtime-x86_64}"
[[ -x "$tool" && -f "$runtime" ]] || { echo 'Place the official appimagetool-x86_64.AppImage and runtime-x86_64 in build/tools/, or set APPIMAGETOOL and APPIMAGE_RUNTIME.' >&2; exit 1; }
appdir="$PWD/build/OxideRF.AppDir"
version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -1)"
artifact="Oxide-RF-$version-x86_64.AppImage"
mkdir -p "$appdir/usr/bin" "$appdir/usr/lib" "$appdir/usr/share/oxide-rf/source" dist
cp target/release/oxide-rf "$appdir/usr/bin/"
cp -L /usr/lib/libOpenCL.so.1 "$appdir/usr/lib/"
cp packaging/AppRun packaging/oxide-rf.desktop "$appdir/"
cp assets/icon.png "$appdir/oxide-rf.png"
chmod +x "$appdir/AppRun" "$appdir/usr/bin/oxide-rf"
cp LICENSE README.md THIRD_PARTY_NOTICES.md "$appdir/usr/share/oxide-rf/"
cp -r src kernels packaging examples assets "$appdir/usr/share/oxide-rf/source/"
cp Cargo.toml Cargo.lock build.rs THIRD_PARTY_NOTICES.md "$appdir/usr/share/oxide-rf/source/"
mkdir -p "$appdir/usr/share/oxide-rf/source/scripts"
cp scripts/build-appimage.sh scripts/benchmark-native.sh "$appdir/usr/share/oxide-rf/source/scripts/"
cp run.sh run.cmd "$appdir/usr/share/oxide-rf/source/"
licenses="$appdir/usr/share/oxide-rf/third-party-licenses"
mkdir -p "$licenses"
registry="${CARGO_HOME:-$HOME/.cargo}/registry/src"
if [[ -d "$registry" ]]; then
    while IFS= read -r -d '' license; do
        crate="$(basename "$(dirname "$license")")"
        mkdir -p "$licenses/$crate"
        cp "$license" "$licenses/$crate/"
    done < <(find "$registry" -mindepth 3 -maxdepth 3 -type f \( -iname 'LICENSE*' -o -iname 'COPYING*' \) -print0)
fi
# ICD loader is bundled, vendor-specific graphics drivers deliberately use the host.
if [[ -d /usr/share/licenses/ocl-icd ]]; then cp -r /usr/share/licenses/ocl-icd "$licenses/"; fi
ARCH=x86_64 "$tool" --appimage-extract-and-run --no-appstream --runtime-file "$runtime" \
    --mksquashfs-opt -processors --mksquashfs-opt 2 \
    "$appdir" "$PWD/dist/$artifact"
cd dist
sha256sum "$artifact" > "$artifact.sha256"
