//! The assistant's way into the app: the same requests as the JSON control channel
//! (`engine.execute`, `engine.commands`, `ui.screenshot`), sent in-process to the UI thread,
//! which runs them between frames (`PhotocraftApp::drain_control`). So the assistant gets the
//! channel's tested semantics (one undo step per `coalesce` key, waiting for background jobs)
//! without a socket or token.

use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::control::{ControlRequest, REPLY_DEADLINE};

/// How long a dispatched request may take (a filter job on a large image) before we stop waiting.
const RUN_LIMIT: Duration = Duration::from_secs(10 * 60);

pub struct Bridge {
    tx: Sender<ControlRequest>,
    ctx: egui::Context,
}

impl Bridge {
    pub fn new(tx: Sender<ControlRequest>, ctx: egui::Context) -> Self {
        Self { tx, ctx }
    }

    /// Sends one request and waits for its `result`, or its error.
    pub fn call(&self, method: &str, params: Value) -> Result<Value, String> {
        let (mut req, rx) = ControlRequest::new(method, params);
        req.deadline = Some(Instant::now() + REPLY_DEADLINE);
        let dispatch = req.dispatch.clone();
        self.tx.send(req).map_err(|_| "PhotoCraft nimmt keine Aufträge mehr an".to_string())?;
        // The UI thread drains requests in its frame loop; wake it.
        self.ctx.request_repaint();
        let reply = rx.recv_timeout(REPLY_DEADLINE + RUN_LIMIT).map_err(|_| dispatch.give_up())?;
        if reply.get("ok").and_then(Value::as_bool) == Some(true) {
            Ok(reply.get("result").cloned().unwrap_or(Value::Null))
        } else {
            Err(reply.get("error").map(|e| e.as_str().map_or_else(|| e.to_string(), str::to_string)).unwrap_or_else(|| "unbekannter Fehler".into()))
        }
    }

    /// `engine.execute`: runs a registry command (waiting for a background job's result).
    pub fn execute(&self, command: &str, params: Value) -> Result<Value, String> {
        self.call("engine.execute", json!({"command": command, "params": params}))
    }

    /// A PNG of the window as base64, for vision models.
    pub fn screenshot_b64(&self) -> Result<String, String> {
        let r = self.call("ui.screenshot", json!({"focus": false}))?;
        r.get("base64").and_then(Value::as_str).map(str::to_string).ok_or_else(|| "Bildschirmfoto ohne Bilddaten".to_string())
    }
}
