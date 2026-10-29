//! The macOS menu bar. Settings and About live here instead of in the
//! window's top bar.

use eframe::egui;
use muda::{
    accelerator::{Accelerator, Code, Modifiers},
    Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem, Submenu,
};

pub enum Action {
    About,
    Settings,
    Quit,
}

pub struct MacMenu {
    // The menu must stay alive for as long as it is shown.
    _menu: Menu,
    about: MenuId,
    settings: MenuId,
    quit: MenuId,
}

impl MacMenu {
    /// Replaces winit's default menu. Call on the main thread once the
    /// application has finished launching.
    pub fn install(ctx: &egui::Context) -> Option<Self> {
        let about = MenuItem::new("About Oxide RF", true, None);
        let settings = MenuItem::new(
            "Settings…",
            true,
            Some(Accelerator::new(Some(Modifiers::META), Code::Comma)),
        );
        // Quit through the window so a running job is stopped first.
        let quit = MenuItem::new(
            "Quit Oxide RF",
            true,
            Some(Accelerator::new(Some(Modifiers::META), Code::KeyQ)),
        );
        let app = Submenu::with_items(
            "Oxide RF",
            true,
            &[
                &about,
                &PredefinedMenuItem::separator(),
                &settings,
                &PredefinedMenuItem::separator(),
                &PredefinedMenuItem::services(None),
                &PredefinedMenuItem::separator(),
                &PredefinedMenuItem::hide(None),
                &PredefinedMenuItem::hide_others(None),
                &PredefinedMenuItem::show_all(None),
                &PredefinedMenuItem::separator(),
                &quit,
            ],
        )
        .ok()?;
        let window = Submenu::with_items(
            "Window",
            true,
            &[
                &PredefinedMenuItem::minimize(None),
                &PredefinedMenuItem::maximize(None),
                &PredefinedMenuItem::separator(),
                &PredefinedMenuItem::bring_all_to_front(None),
            ],
        )
        .ok()?;
        let menu = Menu::with_items(&[&app, &window]).ok()?;
        menu.init_for_nsapp();
        window.set_as_windows_menu_for_nsapp();
        // Menu clicks arrive outside winit's event loop; wake egui to read them.
        let ctx = ctx.clone();
        MenuEvent::set_event_handler(Some(move |event| {
            PENDING.lock().unwrap().push(event);
            ctx.request_repaint();
        }));
        Some(Self {
            _menu: menu,
            about: about.id().clone(),
            settings: settings.id().clone(),
            quit: quit.id().clone(),
        })
    }

    /// Menu items chosen since the last frame.
    pub fn actions(&self) -> Vec<Action> {
        std::mem::take(&mut *PENDING.lock().unwrap())
            .into_iter()
            .filter_map(|event| {
                if event.id == self.about {
                    Some(Action::About)
                } else if event.id == self.settings {
                    Some(Action::Settings)
                } else if event.id == self.quit {
                    Some(Action::Quit)
                } else {
                    None
                }
            })
            .collect()
    }
}

static PENDING: std::sync::Mutex<Vec<MenuEvent>> = std::sync::Mutex::new(Vec::new());
