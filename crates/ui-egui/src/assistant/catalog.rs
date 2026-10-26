//! The command catalog the assistant plans with: the engine registry, searchable by embedding
//! (bge-m3 and similar: multilingual, so German requests find English commands) plus keywords.
//! Only the best matches go into the prompt, which keeps the context small.

use std::collections::HashMap;

use serde_json::Value;

use super::bridge::Bridge;
use super::llm::Llm;
use super::util::{clip, fnv64};

pub struct Cmd {
    pub id: String,
    pub label: String,
    pub menu: Vec<String>,
    pub params: String,
    pub keywords: String,
    pub enabled: bool,
    pub disabled_reason: Option<String>,
}

impl Cmd {
    fn from_json(v: &Value) -> Option<Self> {
        let id = v.get("id")?.as_str()?.to_string();
        Some(Self {
            label: v.get("label").and_then(Value::as_str).unwrap_or("").to_string(),
            menu: v.get("menu").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default(),
            params: v.get("params").and_then(Value::as_str).unwrap_or("{}").to_string(),
            // The registry's English search keywords, plus German terms from the glossary.
            keywords: [v.get("keywords").and_then(Value::as_str).unwrap_or("").to_string(), glossary(&id)]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(", "),
            enabled: v.get("enabled").and_then(Value::as_bool).unwrap_or(true),
            disabled_reason: v.get("disabledReason").and_then(Value::as_str).map(str::to_string),
            id,
        })
    }

    /// The text that is embedded (without params: they dilute the meaning).
    fn embed_text(&self) -> String {
        let mut s = format!("{} | {}", self.id, self.label.trim_end_matches('…'));
        if !self.menu.is_empty() {
            s.push_str(&format!(" | {}", self.menu.join(" > ")));
        }
        if !self.keywords.is_empty() {
            s.push_str(&format!(" | {}", self.keywords));
        }
        s
    }

    /// One prompt line.
    pub fn prompt_line(&self, max_params: usize) -> String {
        let mut s = format!("- {} ({}", self.id, self.label);
        if !self.menu.is_empty() {
            s.push_str(&format!(", {}", self.menu.join(" › ")));
        }
        s.push(')');
        if !self.enabled {
            s.push_str(&format!(" [gerade nicht verfügbar: {}]", self.disabled_reason.as_deref().unwrap_or("?")));
        }
        s.push_str(&format!("\n  params: {}", clip(&self.params, max_params)));
        s
    }
}

/// German terms for commands (part of the id → terms). The registry has only short English names;
/// without these the search barely finds "weichzeichnen" or "Rauschen entfernen".
const GLOSSARY: &[(&str, &str)] = &[
    ("filter.blur.", "weichzeichnen, Weichzeichner, unscharf machen, verwischen, glätten"),
    ("gaussianBlur", "Gaußscher Weichzeichner, weichzeichnen"),
    ("surfaceBlur", "Matter machen, Haut glätten, weichzeichnen mit Kanten"),
    ("lensBlur", "Tiefenunschärfe, Bokeh, Hintergrund unscharf"),
    ("motionBlur", "Bewegungsunschärfe"),
    ("blurGallery", "Weichzeichnergalerie, Tilt-Shift, Hintergrund unscharf"),
    ("filter.sharpen.", "schärfen, Schärfe, scharfzeichnen, Details"),
    ("unsharpMask", "Unscharf maskieren, schärfen"),
    ("highPass", "Hochpass, schärfen, Details"),
    ("reduceNoise", "Rauschen entfernen, Rauschen reduzieren, entrauschen, Bildrauschen"),
    ("filter.noise.median", "Rauschen entfernen, Median"),
    ("despeckle", "Rauschen entfernen, Störungen entfernen"),
    ("dustAndScratches", "Staub und Kratzer entfernen, Rauschen"),
    ("addNoise", "Rauschen hinzufügen, Körnung, Filmkorn"),
    ("brightnessContrast", "Helligkeit, Kontrast erhöhen, Kontrast verringern, heller, dunkler"),
    ("autoContrast", "Kontrast automatisch"),
    ("autoTone", "Tonwert automatisch"),
    ("autoColor", "Farbe automatisch, Farbstich entfernen"),
    ("curves", "Gradationskurven, Kontrast, Helligkeit, Tonwerte, S-Kurve"),
    ("levels", "Tonwertkorrektur, Kontrast, Schwarzpunkt, Weißpunkt"),
    ("exposure", "Belichtung, heller, dunkler"),
    ("shadowsHighlights", "Tiefen/Lichter, Schatten aufhellen, Lichter abdunkeln"),
    ("vibrance", "Dynamik, Sättigung, Farben kräftiger"),
    ("hueSaturation", "Farbton, Sättigung, Helligkeit, Farben kräftiger, entsättigen"),
    ("colorBalance", "Farbbalance, Farbstich, wärmer, kälter, Hautton, Hauttöne"),
    ("photoFilter", "Fotofilter, wärmer, kälter, Farbtemperatur, Wärme, Hautton wärmer"),
    ("selectiveColor", "Selektive Farbkorrektur, Hautton, einzelne Farben ändern"),
    ("blackWhite", "Schwarzweiß, schwarz-weiß"),
    ("desaturate", "entsättigen, Graustufen"),
    ("channelMixer", "Kanalmixer"),
    ("gradientMap", "Verlaufsumsetzung, Color Grading"),
    ("colorLookup", "Farbtabelle, LUT, Look, Color Grading"),
    ("invert", "umkehren, invertieren, Negativ"),
    ("posterize", "Tontrennung"),
    ("threshold", "Schwellenwert"),
    ("cameraRaw", "Camera-Raw-Filter, Temperatur, Weißabgleich, Klarheit, Dunst entfernen, Struktur"),
    ("select.sky", "Himmel auswählen"),
    ("select.subject", "Motiv auswählen, Person auswählen, Objekt freistellen"),
    ("select.object", "Objekt auswählen"),
    ("select.colorRange", "Farbbereich auswählen, Hautton auswählen"),
    ("select.all", "alles auswählen"),
    ("select.deselect", "Auswahl aufheben"),
    ("select.inverse", "Auswahl umkehren"),
    ("select.modify.feather", "weiche Kante, Auswahl weich"),
    ("removeBackground", "Hintergrund entfernen, freistellen"),
    ("spotHealing", "Bereichsreparatur, Flecken entfernen, retuschieren"),
    ("paint.remove", "Entfernen-Werkzeug, Objekt entfernen, retuschieren"),
    ("contentAwareFill", "Inhaltsbasiert füllen, Objekt entfernen"),
    ("redEye", "rote Augen entfernen"),
    ("image.crop", "zuschneiden, beschneiden"),
    ("imageSize", "Bildgröße, verkleinern, vergrößern, skalieren"),
    ("canvasSize", "Arbeitsfläche, Rand hinzufügen"),
    ("imageRotation", "drehen, spiegeln"),
    ("straighten", "gerade ausrichten, Horizont"),
    ("vignette", "Vignette"),
    ("lensCorrection", "Objektivkorrektur, Vignette, Verzeichnung"),
    ("type.create", "Text hinzufügen, Textebene, Beschriftung"),
    ("layer.new.layer", "neue Ebene"),
    ("layer.duplicate", "Ebene duplizieren, Ebene kopieren"),
    ("layer.setProps", "Ebene umbenennen, Deckkraft, Füllmethode, ausblenden"),
    ("layer.layerMask", "Ebenenmaske, Maske"),
    ("layer.layerStyle.dropShadow", "Schlagschatten"),
    ("layer.layerStyle.stroke", "Kontur, Rahmen"),
    ("layer.newFillLayer.solidColor", "Farbfläche, Volltonfarbe"),
    ("layer.newFillLayer.gradient", "Verlauf, Verlaufsebene"),
    ("flattenImage", "auf Hintergrundebene reduzieren, zusammenfügen"),
    ("mergeDown", "Ebenen zusammenführen"),
];

fn glossary(id: &str) -> String {
    GLOSSARY.iter().filter(|(k, _)| id.contains(k)).map(|(_, v)| *v).collect::<Vec<_>>().join(", ")
}

/// Commands the assistant never runs: files, preferences, plug-ins, scripts, app and window
/// control, the system clipboard, preset libraries. The automation authorizer refuses paths
/// outside the automation roots anyway; this is the second gate.
pub fn is_blocked(id: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "app.",
        "plugin.",
        "prefs.",
        "actions.",
        "script",
        "help.",
        "window.",
        "ui.",
        "history.",
        "jobs.",
        "edit.preferences",
        "edit.keyboardShortcuts",
        "edit.menus",
        "edit.toolbar",
        "edit.colorSettings",
        "edit.presets.",
        "edit.purge",
        "edit.undo",
        "edit.redo",
        "edit.stepBackward",
        "edit.stepForward",
        "edit.toggleLastState",
        // The system clipboard.
        "edit.cut",
        "edit.copy",
        "edit.paste",
        // Reading or writing files.
        "layer.exportAs",
        "layer.quickExportAsPng",
        "measurementLog.export",
        "layer.videoLayers.newVideoLayerFromFile",
        "layer.videoLayers.replaceFootage",
        "layer.videoLayers.reloadFrame",
        "pattern.import",
        "pattern.export",
        "gradient.presets.importGrd",
        // Lasting settings and libraries outside the document.
        "edit.defineBrushPreset",
        "edit.defineCustomShape",
        "edit.definePattern",
        "brush.defineFromSelection",
        "pattern.rename",
        "pattern.delete",
        "document.recent.add",
        "type.saveDefaultTypeStyles",
        "type.loadDefaultTypeStyles",
        "layer.layerStyle.makeDefault",
        // Other documents and windows.
        "document.activate",
        "document.move",
        "color.profileMismatch",
    ];
    // Smart objects with linked files or a document window of their own.
    const SMART_OBJECT_FILES: &[&str] = &[
        "exportContents",
        "relinkToFile",
        "saveContents",
        "replaceContents",
        "revealInFinder",
        "convertToLinked",
        "editContents",
        "updateChanged",
        "updateModifiedContent",
        "updateAllModifiedContent",
    ];
    // Preset libraries are only read and applied, never changed.
    const LIBRARY_READ: &[&str] = &["list", "get", "select", "apply", "place", "setCurrent", "use"];
    if id.starts_with("file.") && id != "file.new" {
        return true;
    }
    if id.strip_prefix("layer.smartObjects.").is_some_and(|rest| SMART_OBJECT_FILES.contains(&rest)) {
        return true;
    }
    if (id.contains(".presets.") || id.starts_with("swatches.")) && !LIBRARY_READ.contains(&id.rsplit('.').next().unwrap_or(id)) {
        return true;
    }
    PREFIXES.iter().any(|p| id.starts_with(p))
}

/// Params that would touch files or settings even on an allowed command (`file` on Color Lookup,
/// an ICC path in `profile`). Returns why.
pub fn param_violation(id: &str, params: &Value) -> Option<String> {
    let looks_like_file = |s: &str| {
        let l = s.to_lowercase();
        s.contains(['/', '\\', ':']) || [".icc", ".icm", ".cube", ".3dl", ".look", ".abr", ".pat", ".grd", ".aco", ".ase"].iter().any(|e| l.ends_with(e))
    };
    let m = params.as_object()?;
    if m.contains_key("file") {
        return Some("Parameter `file` (Dateipfad) ist für den Assistenten gesperrt".into());
    }
    for key in ["path", "profile"] {
        if m.get(key).and_then(Value::as_str).is_some_and(looks_like_file) {
            return Some(format!("`{key}` darf kein Dateipfad sein"));
        }
    }
    if id == "edit.checkSpelling" && matches!(m.get("action").and_then(Value::as_str), Some("addToDictionary" | "removeFromDictionary")) {
        return Some("das Benutzerwörterbuch darf der Assistent nicht ändern".into());
    }
    None
}

/// The model's own query: looks up more commands in the catalog (answered here, never sent to
/// the engine).
pub const SEARCH: &str = "befehle.suchen";

/// Pure queries: change nothing and need no undo grouping.
pub fn is_query(id: &str) -> bool {
    matches!(
        id,
        SEARCH
            | "document.inspect"
            | "document.pixel"
            | "document.sampleColor"
            | "session.inspect"
            | "command.list"
            | "type.fonts"
            | "type.hitTest"
            | "edit.installedProfiles"
    ) || id.ends_with(".list")
        || id.ends_with(".get")
        || id.ends_with(".info")
}

/// Always in the prompt, whatever the search finds.
const CORE: &[&str] =
    &["document.inspect", "document.sampleColor", "layer.select", "layer.setProps", "layer.setAdjustment", "layer.duplicate", "layer.new.layer"];

pub struct Catalog {
    pub cmds: Vec<Cmd>,
    vecs: Option<Vec<Vec<f32>>>,
    /// Embedding model the index was built with.
    index_model: String,
    by_id: HashMap<String, usize>,
}

impl Catalog {
    pub fn load(bridge: &Bridge) -> Result<Self, String> {
        let cmds: Vec<Cmd> = commands(bridge)?.iter().filter_map(Cmd::from_json).filter(|c| !is_blocked(&c.id)).collect();
        let by_id = cmds.iter().enumerate().map(|(i, c)| (c.id.clone(), i)).collect();
        Ok(Self { cmds, vecs: None, index_model: String::new(), by_id })
    }

    /// Refreshes only `enabled` (it changes with the document, selection, layer…).
    pub fn refresh(&mut self, bridge: &Bridge) -> Result<(), String> {
        for v in commands(bridge)? {
            if let Some(c) = Cmd::from_json(&v)
                && let Some(cmd) = self.by_id.get(&c.id).copied().and_then(|i| self.cmds.get_mut(i))
            {
                cmd.enabled = c.enabled;
                cmd.disabled_reason = c.disabled_reason;
            }
        }
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&Cmd> {
        self.by_id.get(id).and_then(|&i| self.cmds.get(i))
    }

    pub fn has_index(&self, model: &str) -> bool {
        self.vecs.is_some() && self.index_model == model
    }

    /// Embeds every command (in batches; kept for the session).
    pub fn build_index(&mut self, llm: &Llm, model: &str, progress: &dyn Fn(usize, usize)) -> Result<(), String> {
        let mut vecs = Vec::with_capacity(self.cmds.len());
        let mut cache: HashMap<u64, Vec<f32>> = HashMap::new();
        for chunk in self.cmds.chunks(32) {
            let texts: Vec<String> = chunk.iter().map(Cmd::embed_text).collect();
            let embs = llm.embed(model, &texts)?;
            for (t, e) in texts.iter().zip(embs) {
                cache.insert(fnv64(t), normalize(e));
            }
            progress(cache.len(), self.cmds.len());
        }
        for c in &self.cmds {
            vecs.push(cache.remove(&fnv64(&c.embed_text())).unwrap_or_default());
        }
        self.vecs = Some(vecs);
        self.index_model = model.to_string();
        Ok(())
    }

    /// The `k` best commands for a request, plus the core commands.
    pub fn retrieve(&self, llm: &Llm, query: &str, k: usize) -> Vec<&Cmd> {
        let mut out: Vec<&Cmd> = CORE.iter().filter_map(|id| self.get(id)).collect();
        for c in self.rank(llm, query, k + CORE.len()) {
            if out.len() >= k + CORE.len() {
                break;
            }
            if !out.iter().any(|o| o.id == c.id) {
                out.push(c);
            }
        }
        out
    }

    /// The `k` best search hits alone.
    pub fn rank(&self, llm: &Llm, query: &str, k: usize) -> Vec<&Cmd> {
        let mut scores = vec![0f32; self.cmds.len()];
        if let Some(vecs) = &self.vecs
            && let Ok(e) = llm.embed(&self.index_model, &[query.to_string()])
        {
            let q = normalize(e.into_iter().next().unwrap_or_default());
            for (s, v) in scores.iter_mut().zip(vecs) {
                *s = dot(&q, v);
            }
        }
        let words = tokens(query);
        for (s, c) in scores.iter_mut().zip(&self.cmds) {
            *s += lexical(&words, c);
        }
        let mut idx: Vec<usize> = (0..self.cmds.len()).collect();
        idx.sort_by(|&a, &b| scores.get(b).unwrap_or(&0.0).total_cmp(scores.get(a).unwrap_or(&0.0)));
        idx.into_iter().take(k).filter_map(|i| self.cmds.get(i)).collect()
    }

    /// Suggestions for an unknown command id (a typo or an id the model made up).
    pub fn suggest(&self, wrong: &str, n: usize) -> Vec<&str> {
        let words = tokens(&wrong.replace('.', " "));
        let mut v: Vec<(f32, &str)> = self.cmds.iter().map(|c| (lexical(&words, c), c.id.as_str())).filter(|(s, _)| *s > 0.0).collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
        v.into_iter().take(n).map(|(_, id)| id).collect()
    }
}

fn commands(bridge: &Bridge) -> Result<Vec<Value>, String> {
    match bridge.call("engine.commands", serde_json::json!({}))? {
        Value::Array(a) => Ok(a),
        other => Err(format!("unexpected engine.commands reply: {}", clip(&other.to_string(), 200))),
    }
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 && n.is_finite() {
        v.iter_mut().for_each(|x| *x /= n);
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn tokens(s: &str) -> Vec<String> {
    s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.chars().count() >= 3).map(str::to_string).collect()
}

fn lexical(words: &[String], c: &Cmd) -> f32 {
    let hay = format!("{} {} {}", c.id, c.label, c.keywords).to_lowercase();
    words.iter().filter(|w| hay.contains(w.as_str())).count() as f32 * 0.03
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn blocklist() {
        for id in [
            "file.save",
            "plugin.install",
            "prefs.set",
            "edit.undo",
            "history.selectSnapshot",
            "edit.paste",
            "edit.pasteSpecial.pasteInto",
            "jobs.cancel",
            "layer.quickExportAsPng",
            "layer.smartObjects.relinkToFile",
            "brush.presets.delete",
            "swatches.import",
            "document.presets.save",
            "edit.presets.migratePresets",
        ] {
            assert!(is_blocked(id), "{id}");
        }
        for id in [
            "file.new",
            "layer.newAdjustmentLayer.curves",
            "filter.blur.gaussianBlur",
            "brush.presets.list",
            "style.presets.apply",
            "layer.smartObjects.convertToSmartObject",
        ] {
            assert!(!is_blocked(id), "{id}");
        }
    }

    #[test]
    fn params() {
        assert!(param_violation("layer.newAdjustmentLayer.colorLookup", &json!({"file": "a.cube"})).is_some());
        assert!(param_violation("edit.assignProfile", &json!({"profile": "C:/x/p.icc"})).is_some());
        assert!(param_violation("edit.assignProfile", &json!({"profile": "srgb"})).is_none());
        assert!(param_violation("layer.newAdjustmentLayer.curves", &json!({"path": "work"})).is_none());
        assert!(param_violation("edit.checkSpelling", &json!({"action": "addToDictionary", "word": "x"})).is_some());
        assert!(param_violation("filter.blur.gaussianBlur", &json!({"radius": 2})).is_none());
    }

    #[test]
    fn queries() {
        assert!(is_query("document.inspect"));
        assert!(is_query(SEARCH));
        assert!(is_query("type.info"));
        assert!(!is_query("layer.newAdjustmentLayer.curves"));
    }

    #[test]
    fn every_blocked_registry_command_stays_out_of_reach() {
        // The real registry: nothing that writes files or preferences may slip through.
        for spec in photocraft_engine::commands::command_specs() {
            let id = spec.id;
            if id.starts_with("file.") && id != "file.new" || id.starts_with("prefs.") || id.starts_with("plugin.") {
                assert!(is_blocked(id), "{id}");
            }
        }
    }

    #[test]
    fn glossary_finds_german_terms() {
        assert!(glossary("filter.blur.gaussianBlur").contains("weichzeichnen"));
        assert!(glossary("layer.newAdjustmentLayer.photoFilter").contains("wärmer"));
        assert!(glossary("layer.list").is_empty());
    }
}
