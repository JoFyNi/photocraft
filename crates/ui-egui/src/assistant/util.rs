//! Small text helpers for the assistant: clipping, a stable hash, and pulling the JSON plan out of
//! a model's answer.

use serde_json::Value;

/// At most `max` characters of `s` (cut at a character boundary), with `…` when shortened.
pub fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// FNV-1a: stable across versions, keys the embedding cache.
pub fn fnv64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The first complete JSON object in a model's answer. Drops `<think>…</think>` blocks and
/// Markdown fences, as Qwen and similar models like to add.
pub fn extract_json(text: &str) -> Option<Value> {
    let mut t = text.to_string();
    while let (Some(a), Some(b)) = (t.find("<think>"), t.find("</think>")) {
        if b < a {
            break;
        }
        t.replace_range(a..b + "</think>".len(), "");
    }
    if let Some(rest) = t.find("</think>").and_then(|b| t.get(b + "</think>".len()..)) {
        t = rest.to_string();
    }
    let t = t.trim();
    if let Ok(v @ Value::Object(_)) = serde_json::from_str::<Value>(t) {
        return Some(v);
    }
    // The first balanced {...}, minding strings.
    let mut start = 0;
    while let Some(off) = t.get(start..).and_then(|rest| rest.find('{')) {
        let s = start + off;
        let (mut depth, mut in_str, mut esc) = (0i32, false, false);
        for (i, c) in t.char_indices().skip_while(|(i, _)| *i < s) {
            if in_str {
                match c {
                    _ if esc => esc = false,
                    '\\' => esc = true,
                    '"' => in_str = false,
                    _ => {}
                }
                continue;
            }
            match c {
                '"' => in_str = true,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(Ok(v @ Value::Object(_))) = t.get(s..=i).map(serde_json::from_str::<Value>) {
                            return Some(v);
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
        start = s + 1;
    }
    None
}

/// An `OLLAMA_HOST` value as a URL, the way the Ollama CLI reads it:
/// `192.168.1.5` → `http://192.168.1.5:11434`, `host:8080` → `http://host:8080`.
pub fn ollama_host_url(h: &str) -> String {
    let h = h.trim().trim_end_matches('/');
    let (scheme, rest) = h.split_once("://").unwrap_or(("http", h));
    let host_part = rest.split('/').next().unwrap_or(rest);
    let has_port = if host_part.starts_with('[') { host_part.contains("]:") } else { host_part.contains(':') };
    if has_port {
        format!("{scheme}://{rest}")
    } else {
        let default = if scheme == "https" { 443 } else { 11434 };
        format!("{scheme}://{host_part}:{default}{}", rest.get(host_part.len()..).unwrap_or(""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_host_forms() {
        assert_eq!(ollama_host_url("http://192.168.1.5:11434"), "http://192.168.1.5:11434");
        assert_eq!(ollama_host_url("192.168.1.5"), "http://192.168.1.5:11434");
        assert_eq!(ollama_host_url("192.168.1.5:8080/"), "http://192.168.1.5:8080");
        assert_eq!(ollama_host_url("https://ki.example.org"), "https://ki.example.org:443");
    }

    #[test]
    fn json_from_noisy_answers() {
        let a = "<think>Erst überlegen {nicht das}</think>\n```json\n{\"message\":\"ok\",\"steps\":[],\"done\":true}\n```";
        assert_eq!(extract_json(a).unwrap()["message"], "ok");
        let b = "Plan: {\"message\":\"a } b\",\"done\":false} Danke";
        assert_eq!(extract_json(b).unwrap()["message"], "a } b");
        assert!(extract_json("kein json").is_none());
        // Multi-byte text before the object, and unbalanced braces, never panic.
        assert!(extract_json("äöü {{{").is_none());
        assert_eq!(extract_json("ß→ {\"a\":1}").unwrap()["a"], 1);
    }

    #[test]
    fn clip_respects_chars() {
        assert_eq!(clip("äöüß", 2), "äö…");
        assert_eq!(clip("abc", 5), "abc");
    }
}
