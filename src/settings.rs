//! Remembers the last used GUI settings as JSON in `oxide-rf/settings.json`
//! under the platform's configuration directory (see [`config_dir`]). Settings
//! saved under the former name, `gpu-vhs-decode`, are read once if needed.
use serde_json::{Map, Value};
use std::{fs, path::PathBuf};

/// Settings file location. `VHS_GUI_SETTINGS` overrides it; screenshot test
/// builds only persist when it is set, so fixtures never touch the user's file.
pub fn path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("VHS_GUI_SETTINGS") {
        return Some(path.into());
    }
    if cfg!(feature = "ui-screenshot") {
        return None;
    }
    Some(config_dir()?.join("oxide-rf").join("settings.json"))
}
/// The per-user configuration directory: `%APPDATA%` on Windows,
/// `~/Library/Application Support` on macOS, `$XDG_CONFIG_HOME` or `~/.config`
/// elsewhere.
fn config_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        return std::env::var_os("APPDATA").map(PathBuf::from);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    if cfg!(target_os = "macos") {
        return home.map(|home| home.join("Library").join("Application Support"));
    }
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| home.map(|home| home.join(".config")))
}
/// Settings of the former `gpu-vhs-decode` name; only read, never changed.
fn legacy_path() -> Option<PathBuf> {
    if std::env::var_os("VHS_GUI_SETTINGS").is_some() || cfg!(feature = "ui-screenshot") {
        return None;
    }
    Some(config_dir()?.join("gpu-vhs-decode").join("settings.json"))
}

/// Saved settings, or an empty map when none exist or the file is unreadable.
pub fn load() -> Map<String, Value> {
    path()
        .filter(|path| path.exists())
        .or_else(legacy_path)
        .and_then(|path| fs::read(path).ok())
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|value| match value {
            Value::Object(map) => Some(map),
            _ => None,
        })
        .unwrap_or_default()
}

/// Write atomically, so an interrupted save never leaves a truncated file.
pub fn save(settings: &Value) -> Result<(), String> {
    let Some(path) = path() else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let temporary = path.with_extension("json.tmp");
    let text = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    fs::write(&temporary, text).map_err(|e| e.to_string())?;
    fs::rename(&temporary, &path).map_err(|e| e.to_string())
}

/// Typed readers that ignore missing, mistyped or out-of-range values.
pub struct Reader<'a>(pub &'a Map<String, Value>);
impl Reader<'_> {
    pub fn string(&self, key: &str, target: &mut String) {
        if let Some(value) = self.0.get(key).and_then(Value::as_str) {
            *target = value.into();
        }
    }
    pub fn choice(&self, key: &str, allowed: &[&str], target: &mut String) {
        if let Some(value) = self
            .0
            .get(key)
            .and_then(Value::as_str)
            .filter(|v| allowed.contains(v))
        {
            *target = value.into();
        }
    }
    pub fn flag(&self, key: &str, target: &mut bool) {
        if let Some(value) = self.0.get(key).and_then(Value::as_bool) {
            *target = value;
        }
    }
    pub fn number(&self, key: &str, range: std::ops::RangeInclusive<f64>, target: &mut f64) {
        if let Some(value) = self
            .0
            .get(key)
            .and_then(Value::as_f64)
            .filter(|v| range.contains(v))
        {
            *target = value;
        }
    }
    pub fn count(&self, key: &str, target: &mut u32) {
        if let Some(value) = self
            .0
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
        {
            *target = value;
        }
    }
}
