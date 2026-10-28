//! Fluent-style dark theme and the few composite widgets the GUI is built from:
//! cards, segmented selectors, large action buttons, status pills and metric tiles.
use eframe::egui::{
    self, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Margin, RichText,
    Stroke, TextStyle, Vec2,
};
use std::sync::Arc;

pub const BACKGROUND: Color32 = Color32::from_rgb(28, 28, 31);
pub const PANEL: Color32 = Color32::from_rgb(32, 32, 36);
pub const CARD: Color32 = Color32::from_rgb(41, 41, 46);
pub const CARD_STROKE: Color32 = Color32::from_rgb(56, 56, 63);
pub const FIELD: Color32 = Color32::from_rgb(30, 30, 34);
pub const CONTROL: Color32 = Color32::from_rgb(52, 52, 58);
pub const CONTROL_HOVER: Color32 = Color32::from_rgb(62, 62, 70);
pub const ACCENT: Color32 = Color32::from_rgb(76, 194, 255);
pub const ON_ACCENT: Color32 = Color32::from_rgb(8, 24, 36);
pub const TEXT: Color32 = Color32::from_rgb(236, 236, 240);
pub const MUTED: Color32 = Color32::from_rgb(160, 160, 170);
pub const SUCCESS: Color32 = Color32::from_rgb(108, 203, 95);
pub const WARNING: Color32 = Color32::from_rgb(252, 196, 25);
pub const DANGER: Color32 = Color32::from_rgb(255, 107, 107);
pub const RECORD: Color32 = Color32::from_rgb(229, 72, 77);

const RADIUS: u8 = 8;
/// Family used for headings and button labels.
pub const MEDIUM: &str = "medium";
/// Height shared by buttons, segmented controls and metric chips in a row.
pub const CONTROL_HEIGHT: f32 = 38.0;

/// Load Noto Sans when the system has it (common on Linux desktops); egui's
/// bundled fonts remain as fallback, so nothing is required at runtime.
fn install_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    let read = |names: &[&str]| {
        names.iter().find_map(|name| {
            [
                "/usr/share/fonts/noto",
                "/usr/share/fonts/truetype/noto",
                "/usr/share/fonts/google-noto",
                "/usr/share/fonts/noto-fonts",
                "/usr/share/fonts/TTF",
            ]
            .iter()
            .find_map(|dir| std::fs::read(format!("{dir}/{name}")).ok())
        })
    };
    let mut add = |key: &str, bytes: Option<Vec<u8>>| -> bool {
        let Some(bytes) = bytes else { return false };
        fonts
            .font_data
            .insert(key.into(), Arc::new(FontData::from_owned(bytes)));
        true
    };
    let regular = add("ui-regular", read(&["NotoSans-Regular.ttf"]));
    let medium = add(
        "ui-medium",
        read(&[
            "NotoSans-Medium.ttf",
            "NotoSans-SemiBold.ttf",
            "NotoSans-Bold.ttf",
        ]),
    );
    let mono = add("ui-mono", read(&["NotoSansMono-Regular.ttf"]));
    let families = &mut fonts.families;
    if regular {
        families
            .entry(FontFamily::Proportional)
            .or_default()
            .insert(0, "ui-regular".into());
    }
    if mono {
        families
            .entry(FontFamily::Monospace)
            .or_default()
            .insert(0, "ui-mono".into());
    }
    let mut heading = families
        .get(&FontFamily::Proportional)
        .cloned()
        .unwrap_or_default();
    if medium {
        heading.insert(0, "ui-medium".into());
    }
    families.insert(FontFamily::Name(MEDIUM.into()), heading);
    ctx.set_fonts(fonts);
}

pub fn apply(ctx: &egui::Context) {
    install_fonts(ctx);
    ctx.set_visuals(egui::Visuals::dark());
    ctx.style_mut(|style| {
        let medium = FontFamily::Name(MEDIUM.into());
        style.text_styles = [
            (TextStyle::Heading, FontId::new(22.0, medium.clone())),
            (TextStyle::Body, FontId::proportional(15.0)),
            (TextStyle::Button, FontId::new(15.0, medium)),
            (TextStyle::Small, FontId::proportional(12.5)),
            (TextStyle::Monospace, FontId::monospace(13.0)),
        ]
        .into();
        let spacing = &mut style.spacing;
        spacing.item_spacing = Vec2::new(10.0, 9.0);
        spacing.button_padding = Vec2::new(14.0, 7.0);
        spacing.interact_size = Vec2::new(40.0, 32.0);
        spacing.combo_width = 260.0;
        spacing.text_edit_width = 260.0;
        spacing.icon_width = 18.0;
        spacing.icon_spacing = 8.0;
        spacing.menu_margin = Margin::same(8);
        spacing.scroll.bar_width = 8.0;

        let visuals = &mut style.visuals;
        visuals.panel_fill = PANEL;
        visuals.window_fill = CARD;
        visuals.extreme_bg_color = FIELD;
        visuals.faint_bg_color = CARD;
        visuals.window_corner_radius = CornerRadius::same(12);
        visuals.menu_corner_radius = CornerRadius::same(RADIUS);
        visuals.window_stroke = Stroke::new(1.0_f32, CARD_STROKE);
        visuals.override_text_color = None;
        visuals.hyperlink_color = ACCENT;
        visuals.selection.bg_fill = ACCENT.linear_multiply(0.55);
        visuals.selection.stroke = Stroke::new(1.0_f32, ACCENT);
        visuals.warn_fg_color = WARNING;
        visuals.error_fg_color = DANGER;
        let widgets = &mut visuals.widgets;
        for (state, fill, text) in [
            (&mut widgets.noninteractive, CARD, TEXT),
            (&mut widgets.inactive, CONTROL, TEXT),
            (&mut widgets.hovered, CONTROL_HOVER, TEXT),
            (&mut widgets.active, ACCENT, ON_ACCENT),
            (&mut widgets.open, CONTROL_HOVER, TEXT),
        ] {
            state.corner_radius = CornerRadius::same(RADIUS);
            state.bg_fill = fill;
            state.weak_bg_fill = fill;
            state.fg_stroke = Stroke::new(1.0_f32, text);
            state.expansion = 0.0;
        }
        widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, CARD_STROKE);
        widgets.inactive.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(66, 66, 74));
        widgets.hovered.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(90, 90, 100));
        widgets.active.bg_stroke = Stroke::new(1.0_f32, ACCENT);
        widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, MUTED);
    });
}

/// A titled section with a rounded, slightly raised background.
pub fn card<R>(ui: &mut egui::Ui, title: &str, add_contents: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0_f32, CARD_STROKE))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::symmetric(18, 16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            if !title.is_empty() {
                ui.label(
                    RichText::new(title)
                        .family(FontFamily::Name(MEDIUM.into()))
                        .size(17.0)
                        .color(TEXT),
                );
                ui.add_space(4.0);
            }
            add_contents(ui)
        })
        .inner
}

/// Joined toggle buttons; returns the index clicked this frame, if any.
pub fn segmented(ui: &mut egui::Ui, selected: usize, labels: &[&str]) -> Option<usize> {
    let mut clicked = None;
    egui::Frame::new()
        .fill(FIELD)
        .stroke(Stroke::new(1.0_f32, CARD_STROKE))
        .corner_radius(CornerRadius::same(RADIUS + 2))
        .inner_margin(Margin::same(3))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 3.0;
                // Keep reading order when placed inside a right-to-left layout.
                let mut order: Vec<_> = labels.iter().enumerate().collect();
                if ui.layout().prefer_right_to_left() {
                    order.reverse();
                }
                for (index, label) in order {
                    let active = index == selected;
                    let button = egui::Button::new(RichText::new(*label).color(if active {
                        ON_ACCENT
                    } else {
                        TEXT
                    }))
                    .fill(if active { ACCENT } else { Color32::TRANSPARENT })
                    .stroke(Stroke::NONE)
                    .corner_radius(CornerRadius::same(RADIUS))
                    .min_size(Vec2::new(0.0, 32.0));
                    if ui.add(button).clicked() && !active {
                        clicked = Some(index);
                    }
                }
            });
        });
    clicked
}

/// Large filled call-to-action button.
pub fn action_button(text: &str, fill: Color32, text_color: Color32) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text).size(15.0).color(text_color))
        .fill(fill)
        .stroke(Stroke::NONE)
        .corner_radius(CornerRadius::same(RADIUS + 2))
        .min_size(Vec2::new(120.0, CONTROL_HEIGHT))
}

pub fn primary_button(text: &str) -> egui::Button<'static> {
    action_button(text, ACCENT, ON_ACCENT)
}

pub fn secondary_button(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text).size(15.0))
        .corner_radius(CornerRadius::same(RADIUS + 2))
        .min_size(Vec2::new(120.0, CONTROL_HEIGHT))
}

/// Text in the medium-weight heading family.
pub fn title(text: impl Into<String>, size: f32) -> RichText {
    RichText::new(text.into())
        .size(size)
        .color(TEXT)
        .family(FontFamily::Name(MEDIUM.into()))
}

/// Small rounded status label, e.g. "Recording".
pub fn pill(ui: &mut egui::Ui, text: &str, color: Color32) -> egui::Response {
    egui::Frame::new()
        .fill(color.linear_multiply(0.18))
        .stroke(Stroke::new(1.0_f32, color.linear_multiply(0.6)))
        .corner_radius(CornerRadius::same(255))
        .inner_margin(Margin::symmetric(10, 3))
        .show(ui, |ui| {
            ui.label(
                RichText::new(text)
                    .size(12.5)
                    .color(color)
                    .family(FontFamily::Name(MEDIUM.into())),
            );
        })
        .response
}

/// A labelled live value, e.g. headroom or frames per second.
/// A live value as a one-line chip, as tall as buttons and segmented controls.
pub fn metric(ui: &mut egui::Ui, label: &str, value: &str, color: Color32) {
    egui::Frame::new()
        .fill(FIELD)
        .stroke(Stroke::new(1.0_f32, CARD_STROKE))
        .corner_radius(CornerRadius::same(RADIUS + 2))
        .inner_margin(Margin::symmetric(10, 0))
        .show(ui, |ui| {
            ui.set_min_size(Vec2::new(ui.available_width(), CONTROL_HEIGHT - 2.0));
            ui.horizontal_centered(|ui| {
                ui.label(RichText::new(label).size(12.5).color(MUTED));
                ui.label(
                    RichText::new(value)
                        .size(15.0)
                        .color(color)
                        .family(FontFamily::Name(MEDIUM.into())),
                );
            });
        });
}

/// Secondary explanatory text.
pub fn hint(ui: &mut egui::Ui, text: impl Into<String>) {
    ui.label(RichText::new(text.into()).size(12.5).color(MUTED));
}
/// A hint kept on one line, cut with an ellipsis when it does not fit, so the
/// height of the surrounding layout does not depend on its width.
pub fn hint_line(ui: &mut egui::Ui, text: impl Into<String>) {
    let text = text.into();
    ui.add(egui::Label::new(RichText::new(&text).size(12.5).color(MUTED)).truncate())
        .on_hover_text(text);
}

/// A form row: fixed-width caption followed by the control(s).
pub fn row<R>(ui: &mut egui::Ui, caption: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            Vec2::new(118.0, 32.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.set_min_width(118.0);
                ui.label(RichText::new(caption).color(MUTED));
            },
        );
        add(ui)
    })
    .inner
}

/// Single-line text field that fills the remaining row width. Widgets placed
/// after it must be laid out right-to-left first (see [`field_with_button`]),
/// otherwise the row overflows and a resizable panel keeps growing.
pub fn text_field(ui: &mut egui::Ui, value: &mut String) -> egui::Response {
    ui.add(
        egui::TextEdit::singleline(value)
            .desired_width(ui.available_width())
            .margin(Margin::symmetric(8, 6)),
    )
}

/// Captioned text field with a trailing button; returns true when clicked.
pub fn field_with_button(
    ui: &mut egui::Ui,
    caption: &str,
    value: &mut String,
    button: &str,
) -> bool {
    row(ui, caption, |ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let clicked = ui.button(button).clicked();
            text_field(ui, value);
            clicked
        })
        .inner
    })
}
