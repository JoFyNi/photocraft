//! The agent loop: request → context (document, image values, matching commands) → plan from the
//! model → run it in PhotoCraft → results or errors back to the model → until done.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Map, Value, json};

use super::bridge::Bridge;
use super::catalog::{Catalog, SEARCH, is_blocked, is_query, param_violation};
use super::llm::{Llm, Msg};
use super::util::{clip, extract_json};
use super::{Event, Line};

/// Commands per plan, prompt commands, planning rounds per request.
const MAX_STEPS: usize = 40;
const TOP_K: usize = 30;
const MAX_ROUNDS: usize = 6;
/// Layers listed in the prompt.
const MAX_LAYERS: usize = 80;
/// How deep the layer tree is walked (documents can nest groups deeply).
const MAX_DEPTH: u64 = 32;

const SYSTEM: &str = r#"Du bist der KI-Assistent in PhotoCraft, einer Bildbearbeitung im Stil von Photoshop (Ebenen, Masken, Einstellungsebenen, Filter, Auswahl, Text, Vektoren).
Du bearbeitest Bilder ausschließlich, indem du Befehle der PhotoCraft-Registry planst. Ein Programm führt sie aus und meldet dir Ergebnisse oder Fehler zurück.

Antworte IMMER mit genau einem JSON-Objekt und sonst nichts:
{"message": "kurze Erklärung für den Nutzer, in seiner Sprache",
 "steps": [{"command": "befehls.id", "params": {}}],
 "done": true}

Regeln:
1. Verwende nur Befehls-IDs aus „Verfügbare Befehle“ (oder solche, die dir in dieser Unterhaltung bereits bestätigt wurden). Erfinde keine IDs.
2. Halte dich genau an die Parameterbeschreibung. Schreibweise darin: `"name":typ`, `a..b=d` heißt Bereich a bis b mit Standard d, `?` heißt optional, `|` trennt Alternativen. Lass Parameter weg, die du nicht brauchst.
3. Arbeite nicht-destruktiv: bevorzuge Einstellungsebenen (layer.newAdjustmentLayer.*) und neue Ebenen statt Pixel direkt zu ändern, außer der Nutzer will es ausdrücklich. Nicht jede Korrektur gibt es als Einstellungsebene (nur die in der Liste); für Filter und image.adjustments.* erst die Pixelebene mit layer.duplicate kopieren und die Kopie bearbeiten.
4. Brauchst du erst Informationen (Ebenen, Pixelwerte, Auswahl), plane nur Abfragebefehle und setze "done": false. Du bekommst die Ergebnisse und planst dann weiter. Fehlt dir ein passender Befehl in der Liste, suche weitere mit {"command": "befehle.suchen", "params": {"text": "Stichworte, gern englisch"}} (auch eine Abfrage).
5. Schlägt ein Schritt fehl, bekommst du die Fehlermeldung; korrigiere den Plan (andere Parameter oder anderer Befehl). Bereits erfolgreiche Schritte nicht wiederholen.
6. Setze "done": true, sobald der Auftrag erledigt ist. Ist er mit den Befehlen nicht machbar, erkläre das in "message" und lass "steps" leer.
7. Keine Dateioperationen, keine Einstellungen, keine Plugins – die sind gesperrt.
8. Farbwerte aus document.sampleColor sind Gleitkommazahlen 0..1; die Bildstatistik unten ist in 0..255 angegeben. Koordinaten sind Dokumentpixel.
9. Dosierung: Nutzerwünsche wie „etwas“, „leicht“ bedeuten moderate Werte; „stark“ deutliche Werte. Natürlich wirkende Ergebnisse bevorzugen.
10. Eine neu angelegte Ebene (auch Einstellungsebene, auch eine Kopie aus layer.duplicate) wird sofort aktiv – danach kein layer.select nötig. Filter (filter.*), Malwerkzeuge und image.adjustments.* wirken auf die aktive Ebene und brauchen eine Pixelebene; ist eine Einstellungsebene aktiv, wähle vorher die Pixelebene mit layer.select {"layer": id} aus („layers“ im Dokument). Rate nie die ID einer Ebene, die erst in diesem Plan entsteht: die erfährst du aus dem Ergebnis.
11. Soll eine vorhandene Einstellungsebene stärker/schwächer werden, ändere sie mit layer.setAdjustment {"layer": id, …} statt eine weitere anzulegen.
12. Betrifft der Auftrag nur einen Bildteil (Hintergrund, Himmel, Person, Motiv), wähle ihn vorher aus (z. B. select.subject, select.sky, dann bei Bedarf select.inverse) – sonst wirkt die Änderung auf das ganze Bild. Fehlen dir dafür Befehle, suche sie mit befehle.suchen."#;

/// What one request runs with.
pub struct Opts {
    pub vision: bool,
    pub dry_run: bool,
}

pub struct Agent {
    bridge: Bridge,
    /// Loaded on first use, kept for the session (with its embedding index).
    catalog: Option<Catalog>,
    events: std::sync::mpsc::Sender<Event>,
    /// Earlier requests of this session (for "etwas stärker", "mach das rückgängig und …").
    memory: Vec<String>,
    run_no: u64,
}

impl Agent {
    pub fn new(bridge: Bridge, events: std::sync::mpsc::Sender<Event>) -> Self {
        Self { bridge, catalog: None, events, memory: vec![], run_no: 0 }
    }

    /// Loads the command list and, with an embedding model, its search index (once per model).
    pub fn prepare(&mut self, llm: &Llm, embed_model: &str) -> Result<(), String> {
        if self.catalog.is_none() {
            self.status("Lade die Befehlsliste …");
            self.catalog = Some(Catalog::load(&self.bridge)?);
        }
        let events = self.events.clone();
        let Some(catalog) = self.catalog.as_mut() else { return Err("keine Befehlsliste".into()) };
        if !embed_model.is_empty() && !catalog.has_index(embed_model) {
            let progress = |done: usize, all: usize| {
                let _ = events.send(Event::Status(format!("Indexiere Befehle mit {embed_model}: {done}/{all} (einmalig pro Sitzung) …")));
            };
            if let Err(e) = catalog.build_index(llm, embed_model, &progress) {
                let _ = events.send(Event::Line(Line::Note(format!("Embedding-Suche nicht möglich ({e}), nutze Stichwortsuche."))));
            }
        }
        Ok(())
    }

    pub fn forget(&mut self) {
        self.memory.clear();
    }

    fn say(&self, line: Line) {
        let _ = self.events.send(Event::Line(line));
    }

    fn status(&self, s: &str) {
        let _ = self.events.send(Event::Status(s.to_string()));
    }

    /// Runs one request to the end (or until `stop` is set).
    pub fn run(&mut self, llm: &Llm, prompt: &str, opts: &Opts, stop: &Arc<AtomicBool>) -> Result<(), String> {
        self.run_no += 1;
        let undo_key = format!("assistant-{}-{}", std::process::id(), self.run_no);
        let Some(catalog) = self.catalog.as_mut() else { return Err("keine Befehlsliste".into()) };
        if let Err(e) = catalog.refresh(&self.bridge) {
            let _ = self.events.send(Event::Line(Line::Note(format!("Befehlsstatus nicht aktualisiert: {e}"))));
        }
        let changed = self.plan_and_run(llm, prompt, opts, stop, &undo_key)?;
        self.memory.push(format!("- „{}“ → {}", clip(prompt, 200), if changed.is_empty() { "nichts geändert".to_string() } else { changed.join("; ") }));
        if !changed.is_empty() {
            self.say(Line::Note("Ein Schritt im Protokoll – „Rückgängig“ nimmt den ganzen Auftrag zurück.".into()));
        }
        Ok(())
    }

    /// The rounds of one request; returns the changing steps that ran.
    fn plan_and_run(&self, llm: &Llm, prompt: &str, opts: &Opts, stop: &Arc<AtomicBool>, undo_key: &str) -> Result<Vec<String>, String> {
        let Some(catalog) = self.catalog.as_ref() else { return Err("keine Befehlsliste".into()) };

        // ---------- Context ----------
        self.status("Lese das Dokument …");
        let state = self.doc_state();
        let has_doc = state.get("width").is_some();
        let stats = if has_doc { self.image_stats(&state).ok() } else { None };
        let cmds = catalog.retrieve(llm, prompt, TOP_K);
        let cmd_list = cmds.iter().map(|c| c.prompt_line(700)).collect::<Vec<_>>().join("\n");

        let mut user = String::new();
        if !self.memory.is_empty() {
            user.push_str("Bisherige Aufträge in dieser Sitzung (älteste zuerst):\n");
            for m in self.memory.iter().rev().take(5).rev() {
                user.push_str(&format!("{m}\n"));
            }
            user.push('\n');
        }
        user.push_str(&format!("Auftrag: {prompt}\n\n"));
        if has_doc {
            user.push_str(&format!("Aktuelles Dokument:\n{}\n\n", clip(&state.to_string(), 6000)));
        } else {
            user.push_str("Es ist kein Dokument geöffnet.\n\n");
        }
        if let Some(s) = &stats {
            user.push_str(&format!("Bildstatistik (Stichprobe aus dem Gesamtbild, 0..255):\n{s}\n\n"));
        }
        user.push_str(&format!("Verfügbare Befehle (Auswahl der passendsten):\n{cmd_list}\n"));

        let mut first = Msg::user(user);
        if opts.vision && has_doc {
            match self.bridge.screenshot_b64() {
                Ok(img) => {
                    first.content.push_str("\nEin Bildschirmfoto des PhotoCraft-Fensters ist angehängt.");
                    first.images.push(img);
                }
                Err(e) => self.say(Line::Note(format!("kein Vorschaubild ({e})"))),
            }
        }
        let mut msgs = vec![Msg::system(SYSTEM), first];

        // ---------- Loop ----------
        let mut changed: Vec<String> = vec![];
        let mut parse_retries = 0;
        let mut round = 0;
        while round < MAX_ROUNDS {
            if stop.load(Ordering::Relaxed) {
                self.say(Line::Note("Abgebrochen.".into()));
                break;
            }
            round += 1;
            self.status("Denke nach …");
            let started = std::time::Instant::now();
            let answer = llm.chat(&msgs)?;
            let tokens = llm.last_tokens.get().map(|(p, a)| format!(", {p}+{a} Tokens")).unwrap_or_default();
            self.status(&format!("Antwort nach {:.1} s{tokens}", started.elapsed().as_secs_f32()));
            let Some(plan) = extract_json(&answer) else {
                parse_retries += 1;
                if parse_retries > 2 {
                    return Err(format!("Das Modell liefert kein gültiges JSON: {}", clip(&answer, 300)));
                }
                msgs.push(Msg::assistant(clip(&answer, 2000)));
                msgs.push(Msg::user("Deine Antwort war kein gültiges JSON-Objekt. Antworte nur mit dem JSON im verlangten Format."));
                round -= 1;
                continue;
            };
            let message = plan.get("message").and_then(Value::as_str).unwrap_or("").trim().to_string();
            let done = plan.get("done").and_then(Value::as_bool).unwrap_or(true);
            let steps: Vec<Value> = plan.get("steps").and_then(Value::as_array).cloned().unwrap_or_default();
            if !message.is_empty() {
                self.say(Line::Assistant(message));
            }
            if steps.is_empty() {
                break;
            }
            let steps: Vec<Value> = steps.into_iter().take(MAX_STEPS).collect();
            let only_queries = steps.iter().all(|s| s.get("command").and_then(Value::as_str).is_some_and(is_query));
            if opts.dry_run && !only_queries {
                for s in &steps {
                    self.say(Line::Planned(format!("{} {}", command_of(s), clip(&params_of(s).to_string(), 160))));
                }
                self.say(Line::Note("Probelauf – nichts ausgeführt.".into()));
                break;
            }

            // Run the steps.
            let mut results = vec![];
            let mut had_error = false;
            // After a new layer the model doesn't know its id yet: steps naming a `layer` wait
            // for the feedback rather than hit a guessed layer.
            let mut created_layer = false;
            let mut paused = false;
            for s in &steps {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let cmd = command_of(s).to_string();
                let mut params = params_of(s);
                if created_layer && params.get("layer").is_some() {
                    paused = true;
                    results.push(json!({"command": cmd, "ok": false, "error": "nicht ausgeführt: dieser und alle folgenden Schritte nutzen eine Ebenen-ID, nachdem im selben Plan eine Ebene angelegt wurde. Prüfe die IDs unter „Dokument jetzt“ und plane den Rest neu."}));
                    break;
                }
                let outcome = if cmd.is_empty() {
                    Err("Schritt ohne \"command\"".to_string())
                } else if cmd == SEARCH {
                    let text = params.get("text").and_then(Value::as_str).unwrap_or(prompt);
                    Ok(json!(catalog.rank(llm, text, 10).iter().map(|c| c.prompt_line(700)).collect::<Vec<_>>()))
                } else if is_blocked(&cmd) {
                    Err(format!("`{cmd}` ist für den Assistenten gesperrt"))
                } else if catalog.get(&cmd).is_none() {
                    Err(format!("unbekannter Befehl `{cmd}`. Ähnliche: {}", catalog.suggest(&cmd, 6).join(", ")))
                } else if let Some(why) = param_violation(&cmd, &params) {
                    Err(format!("`{cmd}`: {why}"))
                } else {
                    if !is_query(&cmd)
                        && let Value::Object(m) = &mut params
                    {
                        // One history step per request, across rounds and background jobs.
                        m.insert("coalesce".into(), json!(undo_key));
                    }
                    self.status(&format!("Führe {cmd} aus …"));
                    self.bridge.execute(&cmd, params)
                };
                match outcome {
                    Ok(v) => {
                        created_layer |= creates_layer(&cmd);
                        if !is_query(&cmd) {
                            changed.push(format!("{cmd} {}", clip(&params_of(s).to_string(), 200)));
                        }
                        self.say(Line::Done(cmd.clone()));
                        let max = if cmd == SEARCH { 8000 } else { 1500 };
                        results.push(json!({"command": cmd, "ok": true, "result": clip_value(&v, max)}));
                    }
                    Err(e) => {
                        had_error = true;
                        self.say(Line::Failed(format!("{cmd}: {}", clip(&e, 200))));
                        results.push(json!({"command": cmd, "ok": false, "error": clip(&e, 800)}));
                        break; // later steps often depend on the failed one
                    }
                }
            }
            if done && !had_error && !paused {
                break;
            }
            // Feedback to the model.
            msgs.push(Msg::assistant(serde_json::to_string(&plan).unwrap_or_default()));
            let mut fb = format!("Ergebnisse:\n{}\n", serde_json::to_string_pretty(&results).unwrap_or_default());
            if had_error {
                fb.push_str("Ein Schritt ist fehlgeschlagen; nachfolgende Schritte wurden nicht ausgeführt. Korrigiere und plane den Rest.\n");
            }
            if !only_queries {
                fb.push_str(&format!("Dokument jetzt:\n{}\n", clip(&self.doc_state().to_string(), 4000)));
            }
            if round + 1 == MAX_ROUNDS {
                fb.push_str("Das ist die letzte Runde: schließe ab und setze done=true.\n");
            }
            msgs.push(Msg::user(fb));
        }

        Ok(changed)
    }

    /// The document in brief: size, mode, selection, layers.
    fn doc_state(&self) -> Value {
        let Ok(d) = self.bridge.execute("document.inspect", json!({})) else {
            return json!({});
        };
        let mut out = Map::new();
        for k in ["name", "width", "height", "mode", "depth", "resolution", "activeLayer", "selectedLayers", "hasSelection", "selectionBounds"] {
            if let Some(v) = d.get(k) {
                out.insert(k.into(), v.clone());
            }
        }
        if let Some(Value::Array(tree)) = d.get("layers") {
            let mut rows = vec![];
            flatten_layers(tree, 0, None, &mut rows);
            out.insert("layers".into(), Value::Array(rows));
        }
        Value::Object(out)
    }

    /// Rough image values from a 6×6 grid of averaged samples (for models without vision).
    fn image_stats(&self, state: &Value) -> Result<Value, String> {
        let w = state.get("width").and_then(Value::as_f64).unwrap_or(0.0);
        let h = state.get("height").and_then(Value::as_f64).unwrap_or(0.0);
        if !(w >= 1.0 && h >= 1.0 && w.is_finite() && h.is_finite()) {
            return Err("keine Größe".into());
        }
        let n = 6usize;
        let cell = (w.min(h) / n as f64).floor();
        let size = [101, 51, 31, 11, 5, 3, 1].into_iter().find(|&s| f64::from(s) <= cell).unwrap_or(1);
        let mut px = vec![];
        for gy in 0..n {
            for gx in 0..n {
                let x = ((gx as f64 + 0.5) / n as f64 * w).floor();
                let y = ((gy as f64 + 0.5) / n as f64 * h).floor();
                let v = self.bridge.execute("document.sampleColor", json!({"x": x, "y": y, "size": size, "sampleLayer": "all"}))?;
                let a: Vec<f64> = v.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
                if let [r, g, b, ..] = a.as_slice() {
                    px.push([*r, *g, *b]);
                }
            }
        }
        if px.is_empty() {
            return Err("keine Proben".into());
        }
        let k = px.len() as f64;
        let mean = |i: usize| px.iter().map(|p| p.get(i).copied().unwrap_or(0.0)).sum::<f64>() / k;
        let (r, g, b) = (mean(0), mean(1), mean(2));
        let lum: Vec<f64> = px.iter().map(|[r, g, b]| 0.2126 * r + 0.7152 * g + 0.0722 * b).collect();
        let lm = lum.iter().sum::<f64>() / k;
        let sd = (lum.iter().map(|l| (l - lm).powi(2)).sum::<f64>() / k).sqrt();
        let lmin = lum.iter().copied().fold(f64::INFINITY, f64::min);
        let lmax = lum.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sat = px.iter().map(|p| p.iter().copied().fold(0.0, f64::max) - p.iter().copied().fold(1.0, f64::min)).sum::<f64>() / k;
        let q = |v: f64| (v * 255.0).round();
        // Brightness per row (top → bottom) helps with sky/foreground questions.
        let rows: Vec<f64> = lum.chunks(n).map(|row| q(row.iter().sum::<f64>() / row.len().max(1) as f64)).collect();
        Ok(json!({
            "mittelwertRGB": [q(r), q(g), q(b)],
            "helligkeit": {"mittel": q(lm), "min": q(lmin), "max": q(lmax), "streuung": q(sd)},
            "helligkeitProZeile_obenNachUnten": rows,
            "saettigung_mittel": q(sat),
            "farbstich_R_minus_B": q(r - b),
            "hinweis": format!("{} gemittelte Proben à {size}×{size} px; grobe Schätzung, kein Histogramm", px.len())
        }))
    }
}

/// Commands that make a new layer (its id is only in the result).
fn creates_layer(cmd: &str) -> bool {
    cmd.starts_with("layer.new") || cmd.starts_with("layer.duplicate") || matches!(cmd, "type.create" | "shape.create" | "layer.stampVisible")
}

/// The layer tree of `document.inspect` (topmost first, groups with `children`) as a flat list
/// with depth and parent group; adjustment layers with their kind and values.
fn flatten_layers(tree: &[Value], depth: u64, parent: Option<&Value>, out: &mut Vec<Value>) {
    if depth > MAX_DEPTH {
        return;
    }
    for l in tree {
        if out.len() >= MAX_LAYERS {
            return;
        }
        let mut o = Map::new();
        for k in ["id", "name", "kind", "visible", "bounds", "opacity", "blend", "hasMask", "clipped"] {
            if let Some(v) = l.get(k).filter(|v| !v.is_null()) {
                o.insert(k.into(), v.clone());
            }
        }
        if l.get("selected").and_then(Value::as_bool) == Some(true) {
            o.insert("selected".into(), json!(true));
        }
        if depth > 0 {
            o.insert("depth".into(), json!(depth));
        }
        if let Some(p) = parent {
            o.insert("parent".into(), p.clone());
        }
        if let Some(a) = l.get("adjustment") {
            o.insert("adjustment".into(), clip_value(a, 400));
        }
        out.push(Value::Object(o));
        if let Some(Value::Array(children)) = l.get("children") {
            flatten_layers(children, depth + 1, l.get("id"), out);
        }
    }
}

fn command_of(step: &Value) -> &str {
    step.get("command").and_then(Value::as_str).unwrap_or("")
}

fn params_of(step: &Value) -> Value {
    match step.get("params") {
        Some(Value::Object(m)) => Value::Object(m.clone()),
        _ => json!({}),
    }
}

/// Large results (layer trees) shortened for the prompt.
fn clip_value(v: &Value, max: usize) -> Value {
    let s = v.to_string();
    if s.chars().count() <= max { v.clone() } else { Value::String(clip(&s, max)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_tree_is_flattened_with_depth_and_parent() {
        let tree = json!([
            {"id": 5, "name": "Gruppe", "kind": "Group", "selected": false, "children": [
                {"id": 7, "name": "Kurve", "kind": "Adjustment", "selected": true, "bounds": null, "adjustment": {"kind": "curves"}}
            ]},
            {"id": 2, "name": "Hintergrund", "kind": "Pixel", "bounds": [0, 0, 10, 10]}
        ]);
        let mut rows = vec![];
        flatten_layers(tree.as_array().unwrap(), 0, None, &mut rows);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[1]["id"], 7);
        assert_eq!(rows[1]["depth"], 1);
        assert_eq!(rows[1]["parent"], 5);
        assert_eq!(rows[1]["selected"], true);
        assert!(rows[1].get("bounds").is_none(), "nulls are dropped");
        assert_eq!(rows[2]["bounds"], json!([0, 0, 10, 10]));
    }

    #[test]
    fn deep_layer_trees_are_bounded() {
        let mut tree = json!([{"id": 0, "children": []}]);
        for i in 1..200 {
            tree = json!([{"id": i, "children": tree}]);
        }
        let mut rows = vec![];
        flatten_layers(tree.as_array().unwrap(), 0, None, &mut rows);
        assert!(rows.len() <= MAX_DEPTH as usize + 1);
    }

    #[test]
    fn layer_creating_commands() {
        assert!(creates_layer("layer.duplicate"));
        assert!(creates_layer("layer.newAdjustmentLayer.curves"));
        assert!(!creates_layer("layer.select"));
        assert!(!creates_layer("filter.blur.gaussianBlur"));
    }
}
