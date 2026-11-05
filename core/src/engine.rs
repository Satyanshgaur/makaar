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
        }
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
        let mut png = Vec::new();
        crop.write_to(&mut Cursor::new(&mut png), ImageFormat::Png)?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
        let body = json!({
            "model": "default",
            "temperature": 0,  // greedy: the faithfulness gate forbids sampling
            "max_tokens": max_tokens,
            "logprobs": true,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{b64}")}},
                {"type": "text", "text": (self.prompt)(task)},
            ]}],
        });
        let v: Value = self
            .agent
            .post(&self.url)
            .send_json(&body)?
            .body_mut()
            .read_json()?;
        parse_response(&v, &self.name)
    }
}

fn parse_response(v: &Value, engine: &str) -> Result<Recognition, EngineError> {
    let choice = &v["choices"][0];
    let text = choice["message"]["content"]
        .as_str()
        .ok_or_else(|| EngineError::Response(v.to_string()))?
        .to_string();
    let token_probs = choice["logprobs"]["content"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|t| t["logprob"].as_f64())
                .map(f64::exp)
                .collect()
        })
        .unwrap_or_default();
    Ok(Recognition {
        text,
        token_probs,
        engine: engine.to_string(),
        truncated: choice["finish_reason"] == "length",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_response() {
        let v = json!({"choices": [{"message": {"content": "x^2"}, "finish_reason": "length",
            "logprobs": {"content": [{"logprob": 0.0}, {"logprob": -std::f64::consts::LN_2}]}}]});
        let r = parse_response(&v, "paddle").unwrap();
        assert_eq!(r.text, "x^2");
        assert!((r.confidence() - 0.75).abs() < 1e-9);
        assert!(r.truncated);
        assert!(parse_response(&json!({"error": "x"}), "p").is_err());
        let none =
            parse_response(&json!({"choices": [{"message": {"content": ""}}]}), "p").unwrap();
        assert_eq!(none.confidence(), 1.0);
        assert!(!none.truncated);
    }
}
