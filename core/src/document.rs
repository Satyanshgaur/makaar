//! Rust mirror of `schema/document.schema.json`.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BlockType {
    Text,
    Heading,
    Formula,
    Table,
    Figure,
    List,
    Caption,
    Handwriting,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Block {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: BlockType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub bbox: [f64; 4],
    pub order: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crop: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verified: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flagged: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edited: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Page {
    pub index: u32,
    pub width: u32,
    pub height: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    pub blocks: Vec<Block>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Document {
    pub version: u32,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub pages: Vec<Page>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let j = r#"{"version":1,"id":"j1","pages":[{"index":0,"width":10,"height":10,"blocks":[
            {"id":"b0","type":"formula","bbox":[0,0,1,1],"order":0,"text":"x^2","confidence":0.9}]}]}"#;
        let d: Document = serde_json::from_str(j).unwrap();
        assert_eq!(d.pages[0].blocks[0].kind, BlockType::Formula);
        let back: Document = serde_json::from_str(&serde_json::to_string(&d).unwrap()).unwrap();
        assert_eq!(d, back);
    }

    #[test]
    fn rejects_unknown_type() {
        let j = r#"{"version":1,"id":"j","pages":[{"index":0,"width":1,"height":1,"blocks":[
            {"id":"b","type":"bogus","bbox":[0,0,1,1],"order":0}]}]}"#;
        assert!(serde_json::from_str::<Document>(j).is_err());
    }
}
