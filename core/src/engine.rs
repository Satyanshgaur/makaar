//! Recognition engines. Mirrors `eval/src/makaar_eval/engine.py`: every recognizer, specialist or
//! general model, takes a region crop and a task and returns text plus token probabilities.
use std::io::Cursor;
use std::time::Duration;

use base64::Engine as _;
use image::{ImageFormat, RgbImage};
use serde_json::{json, Value};

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("http: {0}")]
    Http(#[from] ureq::Error),
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    #[error("unexpected response: {0}")]
    Response(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Text,
    Formula,
    Table,
    Chart,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Recognition {
    pub text: String,
    pub token_probs: Vec<f64>,
    pub engine: String,
    /// Generation stopped at the token budget, usually because the model looped.
    pub truncated: bool,
}

impl Recognition {
    /// Mean token probability; 1.0 when the engine reports none.
    pub fn confidence(&self) -> f64 {
        if self.token_probs.is_empty() {
            1.0
        } else {
            self.token_probs.iter().sum::<f64>() / self.token_probs.len() as f64
        }
    }
}

pub trait Engine {
    fn name(&self) -> &str;
    /// `max_tokens` bounds generation so a looping model costs seconds, not minutes.
    fn recognize(
        &mut self,
        crop: &RgbImage,
        task: Task,
        max_tokens: u32,
    ) -> Result<Recognition, EngineError>;

    /// The same reading with the handwriting adapter switched on, if this engine has one.
    fn recognize_with_adapter(
        &mut self,
        _crop: &RgbImage,
        _task: Task,
        _max_tokens: u32,
    ) -> Option<Result<Recognition, EngineError>> {
        None
    }
}

/// PaddleOCR-VL task prompts.
pub fn paddle_prompt(task: Task) -> &'static str {
    match task {
        Task::Text => "OCR:",
        Task::Formula => "Formula Recognition:",
        Task::Table => "Table Recognition:",
        Task::Chart => "Chart Recognition:",
    }
}

/// Any OpenAI-compatible vision server: llama-server (PaddleOCR-VL GGUF) or vLLM.
pub struct OpenAiEngine {
    name: String,
    url: String,
    agent: ureq::Agent,
    prompt: fn(Task) -> &'static str,
    /// The server has a LoRA adapter loaded (id 0) but not applied by default
    /// (`--lora-init-without-apply`); `recognize_with_adapter` switches it on per request.
    pub has_adapter: bool,
}

impl OpenAiEngine {
    pub fn new(name: &str, base_url: &str) -> Self {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(600)))
            .build()
            .into();
        Self {
            name: name.to_string(),
            url: format!("{}/v1/chat/completions", base_url.trim_end_matches('/')),
            agent,
            prompt: paddle_prompt,
            has_adapter: false,
        }
    }

    fn read(&mut self, crop: &RgbImage, task: Task, max_tokens: u32, adapter: bool) -> Result<Recognition, EngineError> {
        let mut png = Vec::new();
        crop.write_to(&mut Cursor::new(&mut png), ImageFormat::Png)?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
        let mut body = json!({
            "model": "default",
            "temperature": 0,  // greedy: the faithfulness gate forbids sampling
            "max_tokens": max_tokens,
            "logprobs": true,
            "stream": true,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{b64}")}},
                {"type": "text", "text": (self.prompt)(task)},
            ]}],
        });
        if self.has_adapter {
            body["lora"] = json!([{"id": 0, "scale": if adapter { 1.0 } else { 0.0 }}]);
        }
        let resp = self.agent.post(&self.url).send_json(&body)?;
        // Streamed so a looping model is cut off early: on CPU, a loop that runs to the token
        // budget costs minutes. Dropping the reader closes the connection, and llama-server
        // cancels the request.
        let reader = std::io::BufReader::new(resp.into_body().into_reader());
        parse_stream(reader, &self.name)
    }
}

impl Engine for OpenAiEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn recognize(
        &mut self,
        crop: &RgbImage,
        task: Task,
        max_tokens: u32,
    ) -> Result<Recognition, EngineError> {
        self.read(crop, task, max_tokens, false)
    }

    fn recognize_with_adapter(
        &mut self,
        crop: &RgbImage,
        task: Task,
        max_tokens: u32,
    ) -> Option<Result<Recognition, EngineError>> {
        self.has_adapter.then(|| {
            self.read(crop, task, max_tokens, true).map(|mut r| {
                r.engine = format!("{}+handwriting-lora", r.engine);
                r
            })
        })
    }
}

/// A repeated tail at least this long (in characters) stops generation.
pub const RUNAWAY_CHARS: usize = 200;

/// True if `text` ends in one unit (1–200 characters) repeated at least 4 times over at least
/// `min_run` characters: the model is looping, not transcribing.
pub fn runaway(text: &str, min_run: usize) -> bool {
    let c: Vec<char> = text.chars().collect();
    let n = c.len();
    (1..=200.min(n / 4)).any(|unit| {
        let run = (unit..n).rev().take_while(|&i| c[i] == c[i - unit]).count() + unit;
        run >= min_run.max(4 * unit)
    }) || counting_loop(text, min_run).is_some()
}

/// The model counting instead of transcribing: `text` ends in one unit repeated at least 8 times
/// over at least `min_run` characters where only a number changes, mostly going up by one
/// (`L_{1} L_{2} L_{3} …`). Returns the character index just after the loop's first unit.
pub fn counting_loop(text: &str, min_run: usize) -> Option<usize> {
    // Digit runs become one placeholder; `at` maps each placeholder char back to `text`.
    let (mut norm, mut at, mut nums) = (vec![], vec![], vec![]);
    let mut chars = text.chars().enumerate().peekable();
    while let Some((i, ch)) = chars.next() {
        at.push(i);
        if ch.is_ascii_digit() {
            let mut v = u64::from(ch as u8 - b'0');
            while let Some((_, d)) = chars.next_if(|(_, d)| d.is_ascii_digit()) {
                v = v.saturating_mul(10).saturating_add(u64::from(d as u8 - b'0'));
            }
            nums.push((norm.len(), v));
            norm.push('\0');
        } else {
            norm.push(ch);
        }
    }
    let n = norm.len();
    // A number at the very end may still be streaming in.
    if text.ends_with(|c: char| c.is_ascii_digit()) {
        nums.pop();
    }
    (2..=200.min(n / 8)).find_map(|unit| {
        if norm[n - unit..].iter().filter(|&&c| c == '\0').count() != 1 {
            return None;
        }
        let run = (unit..n).rev().take_while(|&i| norm[i] == norm[i - unit]).count();
        let first = n - run - unit; // where the periodic tail's first unit starts
        let tail: Vec<u64> = nums.iter().filter(|(p, _)| *p >= first).map(|&(_, v)| v).collect();
        let ups = tail.windows(2).filter(|w| w[0] + 1 == w[1]).count();
        // Mostly +1 steps: the model may restart the count (`L_{99} L_{10} L_{11}`).
        (tail.len() >= 8 && 4 * ups >= 3 * (tail.len() - 1) && n - first >= min_run)
            .then(|| at[first + unit])
    })
}

/// Reads an OpenAI-style server-sent-event stream into a recognition, stopping early (marked
/// `truncated`) when the text starts looping.
pub fn parse_stream(
    reader: impl std::io::BufRead,
    engine: &str,
) -> Result<Recognition, EngineError> {
    let (mut text, mut token_probs, mut truncated) = (String::new(), vec![], false);
    let mut checked = 0;
    for line in reader.lines() {
        let line = line.map_err(|e| EngineError::Response(e.to_string()))?;
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let v: Value = serde_json::from_str(data)
            .map_err(|e| EngineError::Response(format!("{e}: {data}")))?;
        if v.get("error").is_some() {
            return Err(EngineError::Response(v.to_string()));
        }
        let choice = &v["choices"][0];
        if let Some(t) = choice["delta"]["content"].as_str() {
            text.push_str(t);
        }
        if let Some(a) = choice["logprobs"]["content"].as_array() {
            token_probs.extend(a.iter().filter_map(|t| t["logprob"].as_f64()).map(f64::exp));
        }
        if choice["finish_reason"] == "length" {
            truncated = true;
        }
        if text.len() >= checked + 64 {
            checked = text.len();
            if runaway(&text, RUNAWAY_CHARS) {
                truncated = true;
                break;
            }
        }
    }
    Ok(Recognition {
        text,
        token_probs,
        engine: engine.to_string(),
        truncated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse(chunks: &[Value]) -> std::io::Cursor<Vec<u8>> {
        let mut out = String::new();
        for c in chunks {
            out.push_str(&format!("data: {c}\n\n"));
        }
        out.push_str("data: [DONE]\n\n");
        std::io::Cursor::new(out.into_bytes())
    }

    #[test]
    fn parses_stream() {
        let r = parse_stream(
            sse(&[
                json!({"choices": [{"delta": {"content": "x^"}, "logprobs": {"content": [{"logprob": 0.0}]}}]}),
                json!({"choices": [{"delta": {"content": "2"}, "finish_reason": "length",
                    "logprobs": {"content": [{"logprob": -std::f64::consts::LN_2}]}}]}),
            ]),
            "paddle",
        )
        .unwrap();
        assert_eq!(r.text, "x^2");
        assert!((r.confidence() - 0.75).abs() < 1e-9);
        assert!(r.truncated);
        assert!(parse_stream(sse(&[json!({"error": "x"})]), "p").is_err());
        let none = parse_stream(sse(&[]), "p").unwrap();
        assert_eq!(none.confidence(), 1.0);
        assert!(!none.truncated);
    }

    #[test]
    fn stops_a_looping_stream_early() {
        let mut chunks = vec![json!({"choices": [{"delta": {"content": "Proof. "}}]})];
        for _ in 0..500 {
            chunks.push(json!({"choices": [{"delta": {"content": "28 \\le 24 "}}]}));
        }
        let r = parse_stream(sse(&chunks), "p").unwrap();
        assert!(r.truncated);
        assert!(r.text.len() < 400, "stopped after {} chars", r.text.len());
        // ordinary text, or a short legitimate repeat, runs to the end
        assert!(!runaway("0 & 0 & 0 & 0 \\\\ 1 & 0 & 0 & 0", RUNAWAY_CHARS));
        assert!(runaway(&"ab".repeat(150), RUNAWAY_CHARS));
    }

    #[test]
    fn stops_a_counting_stream_early() {
        let mut chunks = vec![json!({"choices": [{"delta": {"content": "w = \\pi n_i\n\n"}}]})];
        for k in 1..300 {
            chunks.push(json!({"choices": [{"delta": {"content": format!("\\( L_{{{k}}} \\)\n\n")}}]}));
        }
        let r = parse_stream(sse(&chunks), "p").unwrap();
        assert!(r.truncated);
        assert!(r.text.len() < 400, "stopped after {} chars", r.text.len());
        // a short numbered list, or a table of numbers, is not a loop
        let list: String = (1..12).map(|k| format!("{k}. item\n")).collect();
        assert!(!runaway(&list, RUNAWAY_CHARS));
        let table: String = (0..40).map(|k| format!("{} & {} \\\\\n", k * 7 % 10, k)).collect();
        assert!(counting_loop(&table, RUNAWAY_CHARS).is_none());
        let rows: String = (1..40).map(|k| format!("{k} & 0.5 \\\\\n")).collect();
        assert!(counting_loop(&rows, RUNAWAY_CHARS).is_none());
    }
}
