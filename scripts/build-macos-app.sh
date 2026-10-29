#!/usr/bin/env bash
# Package the release build as a macOS application bundle with its icon.
set -euo pipefail
cd "$(dirname "$0")/.."
version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -1)"
cargo build --release --locked
app="$PWD/dist/Oxide RF.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp target/release/oxide-rf "$app/Contents/MacOS/"
cp assets/icon.icns "$app/Contents/Resources/oxide-rf.icns"
sed "s/@VERSION@/$version/g" packaging/Info.plist > "$app/Contents/Info.plist"
cp LICENSE THIRD_PARTY_NOTICES.md "$app/Contents/Resources/"
# Ad-hoc signature so Apple silicon runs the bundle; not a Developer ID.
codesign --force --sign - "$app"
cd dist
ditto -c -k --keepParent "Oxide RF.app" "Oxide-RF-$version-macOS.zip"
