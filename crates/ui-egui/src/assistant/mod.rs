//! Filter › Plug-ins › AI Assistant: a floating panel where you describe an edit in your own words
//! and a chat model on your own Ollama server plans it as engine commands.
//!
//! WebAssembly plug-ins can't reach a network (no host imports), so the assistant lives in the
//! shell. A worker thread talks to the model and sends each step to the UI thread as an in-process
//! control request ([`bridge`]), so it runs exactly like an agent on the control channel: one undo
//! step per request (`coalesce`), background jobs awaited, files and preferences blocked
//! ([`catalog::is_blocked`]). Desktop only; the browser build shows a notice.

#[cfg(not(target_arch = "wasm32"))]
mod agent;
#[cfg(not(target_arch = "wasm32"))]
mod bridge;
#[cfg(not(target_arch = "wasm32"))]
mod catalog;
#[cfg(not(target_arch = "wasm32"))]
mod llm;
#[cfg(not(target_arch = "wasm32"))]
mod util;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::menus::MenuItem;
use crate::theme::Tokens;

/// The menu id of Filter › Plug-ins › AI Assistant….
pub const MENU_ID: &str = "plugin.assistant";

/// Server, models and options; saved with the UI state. Empty fields fall back to the environment
/// (`OLLAMA_HOST`, `PHOTOCRAFT_AI_MODEL`, `PHOTOCRAFT_AI_EMBED_MODEL`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AssistantSettings {
    /// Ollama server, e.g. `http://192.168.1.5:11434` (a bare host gets port 11434).
    pub url: String,
    /// Chat model, e.g. `qwen3:32b`.
    pub model: String,
    /// Embedding model for finding commands (`bge-m3`); empty = keyword search only.
    pub embed_model: String,
    /// Let reasoning models think first: better on hard requests, several times slower.
    pub think: bool,
    /// Send a screenshot of the window along (vision models).
    pub vision: bool,
    /// Context length asked of Ollama (its default is too small for the command list).
    pub num_ctx: u32,
}

impl Default for AssistantSettings {
    fn default() -> Self {
        Self { url: String::new(), model: String::new(), embed_model: "bge-m3".into(), think: false, vision: true, num_ctx: 16384 }
    }
}

impl AssistantSettings {
    /// The server URL in effect.
    pub fn server(&self) -> String {
        let configured = if self.url.trim().is_empty() { std::env::var("OLLAMA_HOST").unwrap_or_default() } else { self.url.clone() };
        if configured.trim().is_empty() { "http://localhost:11434".into() } else { host_url(&configured) }
    }

    /// The chat model in effect, if any.
    pub fn chat_model(&self) -> Option<String> {
        let m = if self.model.trim().is_empty() { std::env::var("PHOTOCRAFT_AI_MODEL").unwrap_or_default() } else { self.model.clone() };
        Some(m.trim().to_string()).filter(|m| !m.is_empty())
    }

    /// The embedding model in effect ("" = keywords only).
    pub fn embedding_model(&self) -> String {
        let m = if self.embed_model.trim().is_empty() { std::env::var("PHOTOCRAFT_AI_EMBED_MODEL").unwrap_or_default() } else { self.embed_model.clone() };
        m.trim().to_string()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn host_url(h: &str) -> String {
    util::ollama_host_url(h)
}

#[cfg(target_arch = "wasm32")]
fn host_url(h: &str) -> String {
    h.trim().to_string()
}

/// One line of the conversation.
#[derive(Clone, Debug, PartialEq)]
pub enum Line {
    User(String),
    Assistant(String),
    /// A step shown in a dry run.
    Planned(String),
    Done(String),
    Failed(String),
    Note(String),
}

/// What the worker reports.
pub enum Event {
    Line(Line),
    Status(String),
    /// The request ended, with its error if it failed.
    Finished(Option<String>),
}

/// The panel's session state (not saved).
#[derive(Default)]
pub struct Assistant {
    pub lines: Vec<Line>,
    pub input: String,
    pub status: String,
    pub busy: bool,
    /// Plan only, change nothing.
    pub dry_run: bool,
    pub show_settings: bool,
    /// The server's models, after "Load models".
    pub models: Vec<String>,
    /// Control requests from the worker, drained with the control channel's.
    pub(crate) requests: Option<std::sync::mpsc::Receiver<crate::control::ControlRequest>>,
    #[cfg(not(target_arch = "wasm32"))]
    worker: Option<Worker>,
}

#[cfg(not(target_arch = "wasm32"))]
struct Worker {
    jobs: std::sync::mpsc::Sender<Job>,
    events: std::sync::mpsc::Receiver<Event>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(not(target_arch = "wasm32"))]
enum Job {
    Run { prompt: String, settings: AssistantSettings, dry_run: bool },
    Forget,
}

/// Inserts "AI Assistant…" into Filter › Plug-ins, above "Install Plug-in…".
pub fn insert_menu_item(items: &mut Vec<MenuItem>) {
    let Some(at) = items.iter().position(|i| i.id == "plugin.install") else { return };
    let Some(path) = items.get(at).map(|i| i.path.clone()) else { return };
    let item = MenuItem { id: MENU_ID.into(), label: "AI Assistant…".into(), path: path.clone(), shortcut: None, enabled: true, checked: None, color: None };
    let separator = MenuItem { id: "---".into(), label: "---".into(), path, shortcut: None, enabled: false, checked: None, color: None };
    items.insert(at, separator);
    items.insert(at, item);
}

/// Opens or closes the panel from its menu item.
pub fn menu(app: &mut PhotocraftApp, id: &str) -> Option<Result<Value, String>> {
    if id != MENU_ID {
        return None;
    }
    app.ui.panels.assistant = !app.ui.panels.assistant;
    Some(Ok(json!({"open": app.ui.panels.assistant})))
}

/// Starts a request (from the panel's Send button).
#[cfg(not(target_arch = "wasm32"))]
fn send(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let prompt = app.assistant.input.trim().to_string();
    if prompt.is_empty() || app.assistant.busy {
        return;
    }
    if app.ui.assistant.chat_model().is_none() {
        app.assistant.lines.push(Line::Failed(tl!("Enter a chat model in the settings (gear button).").to_string()));
        app.assistant.show_settings = true;
        return;
    }
    let worker = app.assistant.worker.get_or_insert_with(|| spawn(&mut app.assistant.requests, ctx));
    worker.stop.store(false, std::sync::atomic::Ordering::Relaxed);
    let job = Job::Run { prompt: prompt.clone(), settings: app.ui.assistant.clone(), dry_run: app.assistant.dry_run };
    if worker.jobs.send(job).is_err() {
        app.assistant.worker = None;
        app.assistant.lines.push(Line::Failed(tl!("The assistant stopped; send again to restart it.").to_string()));
        return;
    }
    app.assistant.lines.push(Line::User(prompt));
    app.assistant.input.clear();
    app.assistant.busy = true;
    app.assistant.status = tl!("Starting…").to_string();
}

/// Starts the worker thread and its in-process request channel.
#[cfg(not(target_arch = "wasm32"))]
fn spawn(requests: &mut Option<std::sync::mpsc::Receiver<crate::control::ControlRequest>>, ctx: &egui::Context) -> Worker {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc::channel;

    let (req_tx, req_rx) = channel();
    *requests = Some(req_rx);
    let (job_tx, job_rx) = channel::<Job>();
    let (event_tx, event_rx) = channel::<Event>();
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let ctx = ctx.clone();
    let spawned = std::thread::Builder::new().name("ai-assistant".into()).spawn(move || {
        let mut agent = agent::Agent::new(bridge::Bridge::new(req_tx, ctx.clone()), event_tx.clone());
        while let Ok(job) = job_rx.recv() {
            match job {
                Job::Forget => agent.forget(),
                Job::Run { prompt, settings, dry_run } => {
                    let result = run_job(&mut agent, &prompt, &settings, dry_run, &thread_stop);
                    let _ = event_tx.send(Event::Finished(result.err()));
                }
            }
            ctx.request_repaint();
        }
    });
    if let Err(e) = spawned {
        log::error!("AI assistant: cannot start its thread: {e}");
    }
    Worker { jobs: job_tx, events: event_rx, stop }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_job(
    agent: &mut agent::Agent,
    prompt: &str,
    settings: &AssistantSettings,
    dry_run: bool,
    stop: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<(), String> {
    let model = settings.chat_model().ok_or("kein Chat-Modell eingestellt")?;
    let llm = llm::Llm::new(&settings.server(), &model, settings.num_ctx.clamp(2048, 262_144), settings.think);
    agent.prepare(&llm, &settings.embedding_model())?;
    agent.run(&llm, prompt, &agent::Opts { vision: settings.vision, dry_run }, stop)
}

/// Takes the worker's reports into the panel (UI thread, every frame).
fn poll(app: &mut PhotocraftApp) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let Some(worker) = app.assistant.worker.as_ref() else { return };
        let mut ended = false;
        while let Ok(ev) = worker.events.try_recv() {
            match ev {
                Event::Line(l) => app.assistant.lines.push(l),
                Event::Status(s) => app.assistant.status = s,
                Event::Finished(err) => {
                    if let Some(e) = err {
                        app.assistant.lines.push(Line::Failed(e));
                    }
                    ended = true;
                }
            }
        }
        if ended {
            app.assistant.busy = false;
            app.assistant.status.clear();
        }
    }
    #[cfg(target_arch = "wasm32")]
    let _ = app;
}

/// The floating panel.
pub fn window(app: &mut PhotocraftApp, ctx: &egui::Context) {
    poll(app);
    if !app.ui.panels.assistant {
        return;
    }
    let t = Tokens::get(ctx);
    let frame = egui::Frame::NONE
        .fill(t.card)
        .stroke(egui::Stroke::new(1.0, t.card_border))
        .corner_radius(egui::CornerRadius::same(t.radius_lg as u8))
        .shadow(egui::Shadow { offset: [0, 10], blur: 30, spread: 0, color: t.shadow })
        .inner_margin(egui::Margin::same(10));
    let canvas = app.last_canvas_rect;
    let mut open = true;
    egui::Window::new(tl!("AI Assistant"))
        .id(egui::Id::unique("ai-assistant-panel"))
        .title_bar(false)
        .resizable(true)
        .frame(frame)
        .default_size([360.0, 460.0])
        .min_width(280.0)
        .default_pos(egui::pos2(canvas.left() + 24.0, canvas.top() + 24.0))
        .show(ctx, |ui| body(app, ui, &mut open));
    if !open {
        app.ui.panels.assistant = false;
    }
}

fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, open: &mut bool) {
    let t = Tokens::get(ui.ctx());
    ui.horizontal(|ui| {
        ui.strong(tl!("AI Assistant"));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if crate::icons::button(ui, "x", 20.0, false, tl!("Close")).clicked() {
                *open = false;
            }
            if crate::icons::button(ui, "settings", 20.0, app.assistant.show_settings, tl!("Settings")).clicked() {
                app.assistant.show_settings = !app.assistant.show_settings;
            }
        });
    });
    crate::widgets::hairline(ui);
    if cfg!(target_arch = "wasm32") {
        ui.label(tl!("The AI assistant needs the desktop app: the browser can't reach your AI server."));
        return;
    }
    if app.assistant.show_settings {
        settings(app, ui);
        crate::widgets::hairline(ui);
    }

    // The conversation.
    let input_height = 64.0;
    let avail = (ui.available_height() - input_height - 34.0).max(80.0);
    egui::ScrollArea::vertical().id_salt("ai-assistant-log").max_height(avail).auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
        if app.assistant.lines.is_empty() {
            ui.label(
                egui::RichText::new(tl!(
                    "Describe what should happen, e.g. “Make the image warmer and a little more contrasty”. Each request is one step in the History panel."
                ))
                .color(t.text_dim),
            );
        }
        for line in &app.assistant.lines {
            let (prefix, text, color) = match line {
                Line::User(s) => ("› ", s, t.text),
                Line::Assistant(s) => ("", s, t.text),
                Line::Planned(s) => ("· ", s, t.text_dim),
                Line::Done(s) => ("✓ ", s, t.text_dim),
                Line::Failed(s) => ("✗ ", s, t.danger),
                Line::Note(s) => ("", s, t.text_dim),
            };
            let rich = egui::RichText::new(format!("{prefix}{text}")).color(color);
            let rich = if matches!(line, Line::User(_)) { rich.strong() } else { rich };
            ui.add(egui::Label::new(rich).wrap());
        }
        if app.assistant.busy {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(egui::RichText::new(&app.assistant.status).color(t.text_dim));
            });
        }
    });

    // Input and actions.
    let edit = egui::TextEdit::multiline(&mut app.assistant.input).desired_rows(2).desired_width(f32::INFINITY).hint_text(tl!("What should I do?"));
    let response = ui.add_enabled(!app.assistant.busy, edit);
    let enter = response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
    #[cfg(not(target_arch = "wasm32"))]
    ui.horizontal(|ui| {
        {
            if app.assistant.busy {
                if ui.button(tl!("Stop")).clicked()
                    && let Some(w) = app.assistant.worker.as_ref()
                {
                    w.stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    app.assistant.status = tl!("Stopping after this step…").to_string();
                }
            } else if ui.button(tl!("Send")).clicked() || enter {
                // Enter inserted a newline into the field; it isn't part of the request.
                let trimmed = app.assistant.input.trim_end_matches(['\n', '\r']).to_string();
                app.assistant.input = trimmed;
                send(app, ui.ctx());
            }
            let can_undo = !app.assistant.busy && app.session.is_enabled("edit.undo");
            if ui.add_enabled(can_undo, egui::Button::new(tl!("Undo"))).on_hover_text(tl!("Undo the last request (one History step)")).clicked() {
                let _ = crate::menus::invoke(app, ui.ctx(), "edit.undo", json!({}));
            }
            ui.checkbox(&mut app.assistant.dry_run, tl!("Plan only"));
            if ui.add_enabled(!app.assistant.busy, egui::Button::new(tl!("New chat"))).clicked() {
                app.assistant.lines.clear();
                if let Some(w) = app.assistant.worker.as_ref() {
                    let _ = w.jobs.send(Job::Forget);
                }
            }
        }
    });
    #[cfg(target_arch = "wasm32")]
    let _ = enter;
}

fn settings(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let s = &mut app.ui.assistant;
    egui::Grid::new("ai-assistant-settings").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
        ui.label(tl!("AI server"));
        let server = s.server();
        ui.add(egui::TextEdit::singleline(&mut s.url).hint_text(server).desired_width(f32::INFINITY));
        ui.end_row();
        ui.label(tl!("Chat model"));
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut s.model).hint_text("qwen3:32b").desired_width(150.0));
            if !app.assistant.models.is_empty() {
                egui::ComboBox::from_id_salt("ai-assistant-models").selected_text("▾").width(24.0).show_ui(ui, |ui| {
                    for m in &app.assistant.models {
                        if ui.selectable_label(s.model == *m, m).clicked() {
                            s.model = m.clone();
                        }
                    }
                });
            }
        });
        ui.end_row();
        ui.label(tl!("Embedding model"));
        ui.add(egui::TextEdit::singleline(&mut s.embed_model).hint_text("bge-m3").desired_width(f32::INFINITY));
        ui.end_row();
        ui.label(tl!("Context length"));
        ui.add(egui::DragValue::new(&mut s.num_ctx).range(2048..=262_144).speed(256));
        ui.end_row();
    });
    ui.checkbox(&mut s.think, tl!("Let the model think first (slower)"));
    ui.checkbox(&mut s.vision, tl!("Send a screenshot along (vision models)"));
    #[cfg(not(target_arch = "wasm32"))]
    if ui.button(tl!("Load models")).on_hover_text(tl!("Ask the server which models it has")).clicked() {
        match llm::Llm::new(&s.server(), "", 0, false).models() {
            Ok(models) => {
                app.assistant.status = String::new();
                app.assistant.lines.push(Line::Note(format!("{}: {}", tl!("Models on the server"), models.join(", "))));
                app.assistant.models = models;
            }
            Err(e) => app.assistant.lines.push(Line::Failed(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_fall_back_and_normalise_the_server() {
        let s = AssistantSettings { url: "192.168.1.5".into(), ..Default::default() };
        if !cfg!(target_arch = "wasm32") {
            assert_eq!(s.server(), "http://192.168.1.5:11434");
        }
        let s = AssistantSettings { model: "  qwen3:32b ".into(), ..Default::default() };
        assert_eq!(s.chat_model().as_deref(), Some("qwen3:32b"));
    }

    #[test]
    fn old_ui_state_without_assistant_settings_still_loads() {
        let s: AssistantSettings = serde_json::from_value(json!({"model": "m"})).unwrap();
        assert_eq!(s.model, "m");
        assert_eq!(s.embed_model, "bge-m3");
        assert_eq!(s.num_ctx, 16384);
    }

    #[test]
    fn menu_item_sits_above_install_plugin() {
        let mut items = vec![MenuItem {
            id: "plugin.install".into(),
            label: "Install Plug-in…".into(),
            path: vec!["Filter".into(), "Plug-ins".into()],
            shortcut: None,
            enabled: true,
            checked: None,
            color: None,
        }];
        insert_menu_item(&mut items);
        assert_eq!(items.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(), [MENU_ID, "---", "plugin.install"]);
        assert_eq!(items[0].path, ["Filter", "Plug-ins"]);
    }
}
