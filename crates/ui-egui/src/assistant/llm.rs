//! Client for the user's own Ollama server (`/api/chat`, `/api/embed`): the chat model plans, the
//! embedding model finds commands. Desktop only; the browser build has no assistant.

use std::cell::Cell;
use std::time::Duration;

use serde_json::{Value, json};

use super::util::clip;

pub struct Msg {
    pub role: &'static str,
    pub content: String,
    /// PNG images as base64 (vision models only).
    pub images: Vec<String>,
}

impl Msg {
    pub fn system(c: impl Into<String>) -> Self {
        Self { role: "system", content: c.into(), images: vec![] }
    }
    pub fn user(c: impl Into<String>) -> Self {
        Self { role: "user", content: c.into(), images: vec![] }
    }
    pub fn assistant(c: impl Into<String>) -> Self {
        Self { role: "assistant", content: c.into(), images: vec![] }
    }
}

pub struct Llm {
    base: String,
    pub model: String,
    num_ctx: u32,
    /// Qwen3-style thinking: off is several times faster with the same plans.
    think: bool,
    agent: ureq::Agent,
    /// Tokens of the last chat answer (prompt, answer).
    pub last_tokens: Cell<Option<(u64, u64)>>,
}

impl Llm {
    pub fn new(base: &str, model: &str, num_ctx: u32, think: bool) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(15)))
            // Large models on one's own hardware can take minutes.
            .timeout_global(Some(Duration::from_secs(20 * 60)))
            .http_status_as_error(false)
            .build()
            .into();
        Self { base: base.trim().trim_end_matches('/').to_string(), model: model.trim().to_string(), num_ctx, think, agent, last_tokens: Cell::new(None) }
    }

    pub fn chat(&self, msgs: &[Msg]) -> Result<String, String> {
        let messages: Vec<Value> = msgs
            .iter()
            .map(|m| {
                let mut o = json!({"role": m.role, "content": m.content});
                if !m.images.is_empty() {
                    o["images"] = json!(m.images);
                }
                o
            })
            .collect();
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
            "format": "json",
            "options": {"temperature": 0.2, "num_ctx": self.num_ctx},
        });
        if !self.think {
            body["think"] = json!(false);
        }
        let v = self.post("/api/chat", &body)?;
        let count = |k: &str| v.get(k).and_then(Value::as_u64);
        self.last_tokens.set(count("prompt_eval_count").zip(count("eval_count")));
        v.pointer("/message/content")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("Antwort ohne message.content: {}", clip(&v.to_string(), 300)))
    }

    /// Embeddings for several texts.
    pub fn embed(&self, model: &str, inputs: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let v = self.post("/api/embed", &json!({"model": model, "input": inputs}))?;
        let out: Vec<Vec<f32>> = v
            .get("embeddings")
            .and_then(Value::as_array)
            .ok_or("Antwort ohne embeddings")?
            .iter()
            .filter_map(|a| a.as_array().map(|x| x.iter().filter_map(Value::as_f64).map(|f| f as f32).collect()))
            .collect();
        if out.len() != inputs.len() {
            return Err(format!("{} Embeddings für {} Texte erhalten", out.len(), inputs.len()));
        }
        Ok(out)
    }

    /// The server's installed models (`/api/tags`), for the settings' model picker. Asked on the
    /// UI thread, so it gives up after a few seconds.
    pub fn models(&self) -> Result<Vec<String>, String> {
        let url = format!("{}/api/tags", self.base);
        let quick: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(4))).build().into();
        let mut resp = quick.get(&url).call().map_err(|e| format!("KI-Server {url} nicht erreichbar: {e}"))?;
        let text = resp.body_mut().read_to_string().map_err(|e| format!("ungültige Antwort von {url}: {e}"))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("ungültige Antwort von {url}: {e}"))?;
        Ok(v.get("models").and_then(Value::as_array).into_iter().flatten().filter_map(|m| m.get("name").and_then(Value::as_str)).map(str::to_string).collect())
    }

    fn post(&self, path: &str, body: &Value) -> Result<Value, String> {
        let url = format!("{}{}", self.base, path);
        let mut resp = self
            .agent
            .post(&url)
            .header("Content-Type", "application/json")
            .send(body.to_string())
            .map_err(|e| format!("KI-Server {url} nicht erreichbar: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.body_mut().read_to_string().unwrap_or_default();
            return Err(format!("KI-Server {url} antwortet {}: {}", status.as_u16(), clip(&text, 400)));
        }
        let text = resp.body_mut().with_config().limit(64 * 1024 * 1024).read_to_string().map_err(|e| format!("ungültige Antwort von {url}: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("ungültiges JSON von {url}: {e}"))
    }
}
