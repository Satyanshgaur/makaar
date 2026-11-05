//! Model files: where they live, whether they're present, and first-run download from Hugging
//! Face pinned to a commit and checked against SHA-256.
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("download {0}: {1}")]
    Http(String, String),
    #[error("{0}: SHA-256 mismatch (got {1}); the partial file was removed")]
    Checksum(String, String),
}

/// One file of a model, fetched from `https://huggingface.co/{repo}/resolve/{rev}/{file}`.
#[derive(Debug, Clone, Copy)]
pub struct ModelFile {
    pub repo: &'static str,
    pub rev: &'static str,
    pub file: &'static str,
    /// Path under the models directory.
    pub dest: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

pub const LAYOUT: ModelFile = ModelFile {
    repo: "PaddlePaddle/PP-DocLayoutV3_onnx",
    rev: "46bbdf188bb0a772c08aed74882ce7e51a8f1ea6",
    file: "inference.onnx",
    dest: "pp-doclayoutv3/inference.onnx",
    sha256: "45bf71750b00739a41fc209f132eb104a4d6b5bb29483c9078164d8b87cf28ba",
    size: 130_502_049,
};

pub const PADDLE_VL: ModelFile = ModelFile {
    repo: "PaddlePaddle/PaddleOCR-VL-1.6-GGUF",
    rev: "511b09642bb324401f15f97cc23bc67e8f0a291d",
    file: "PaddleOCR-VL-1.6-GGUF.gguf",
    dest: "paddleocr-vl-1.6/PaddleOCR-VL-1.6-GGUF.gguf",
    sha256: "f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8",
    size: 935_769_056,
};

pub const PADDLE_VL_MMPROJ: ModelFile = ModelFile {
    repo: "PaddlePaddle/PaddleOCR-VL-1.6-GGUF",
    rev: "511b09642bb324401f15f97cc23bc67e8f0a291d",
    file: "PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
    dest: "paddleocr-vl-1.6/PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
    sha256: "204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a",
    size: 881_770_560,
};

pub const OCR_DET: ModelFile = ModelFile {
    repo: "PaddlePaddle/PP-OCRv6_small_det_onnx",
    rev: "28fe5895c24fd108c19eb3e8479f4ab385fbfc62",
    file: "inference.onnx",
    dest: "pp-ocrv6-small-det/inference.onnx",
    sha256: "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e",
    size: 9_880_512,
};

pub const OCR_REC: ModelFile = ModelFile {
    repo: "PaddlePaddle/PP-OCRv6_small_rec_onnx",
    rev: "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
    file: "inference.onnx",
    dest: "pp-ocrv6-small-rec/inference.onnx",
    sha256: "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
    size: 21_159_378,
};

/// The recognizer's character dictionary.
pub const OCR_REC_DICT: ModelFile = ModelFile {
    repo: "PaddlePaddle/PP-OCRv6_small_rec_onnx",
    rev: "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
    file: "inference.yml",
    dest: "pp-ocrv6-small-rec/inference.yml",
    sha256: "ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1",
    size: 150_579,
};

/// Everything `makaar convert` needs.
pub const REQUIRED: [ModelFile; 6] = [
    LAYOUT,
    OCR_DET,
    OCR_REC,
    OCR_REC_DICT,
    PADDLE_VL,
    PADDLE_VL_MMPROJ,
];

/// `$MAKAAR_MODELS`, else `./models` if it holds the layout model (a source checkout), else the
/// per-user data directory.
pub fn default_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("MAKAAR_MODELS") {
        return d.into();
    }
    let local = PathBuf::from("models");
    if local.join(LAYOUT.dest).exists() {
        return local;
    }
    let data = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    };
    data.unwrap_or_else(|| ".".into()).join("makaar/models")
}

impl ModelFile {
    pub fn path(&self, dir: &Path) -> PathBuf {
        dir.join(self.dest)
    }

    pub fn url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.rev, self.file
        )
    }

    /// Present with the expected size (cheap; `verify` hashes).
    pub fn present(&self, dir: &Path) -> bool {
        std::fs::metadata(self.path(dir)).is_ok_and(|m| m.len() == self.size)
    }

    pub fn verify(&self, dir: &Path) -> Result<bool, ModelError> {
        Ok(sha256_file(&self.path(dir))? == self.sha256)
    }

    /// Downloads to `<dest>.part`, hashing as it goes, and renames on a checksum match.
    /// `progress(done, total)` is called every few MB.
    pub fn download(
        &self,
        dir: &Path,
        mut progress: impl FnMut(u64, u64),
    ) -> Result<(), ModelError> {
        let dest = self.path(dir);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let part = dest.with_extension("part");
        let resp = ureq::get(&self.url())
            .call()
            .map_err(|e| ModelError::Http(self.file.into(), e.to_string()))?;
        let mut body = resp.into_body().into_reader();
        let mut out = std::fs::File::create(&part)?;
        let mut hash = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let (mut done, mut last) = (0u64, 0u64);
        loop {
            let n = body.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hash.update(&buf[..n]);
            out.write_all(&buf[..n])?;
            done += n as u64;
            if done - last >= 8 << 20 {
                progress(done, self.size);
                last = done;
            }
        }
        out.sync_all()?;
        progress(done, self.size);
        let got = hex(&hash.finalize());
        if got != self.sha256 {
            let _ = std::fs::remove_file(&part);
            return Err(ModelError::Checksum(self.file.into(), got));
        }
        std::fs::rename(part, dest)?;
        Ok(())
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_file(path: &Path) -> Result<String, ModelError> {
    let mut f = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    std::io::copy(&mut f, &mut hash)?;
    Ok(hex(&hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_and_presence() {
        let dir = std::env::temp_dir().join(format!("makaar-models-{}", std::process::id()));
        let f = ModelFile {
            dest: "x/a.bin",
            size: 3,
            sha256: "",
            ..LAYOUT
        };
        assert!(!f.present(&dir));
        std::fs::create_dir_all(dir.join("x")).unwrap();
        std::fs::write(f.path(&dir), b"abc").unwrap();
        assert!(f.present(&dir));
        assert_eq!(
            sha256_file(&f.path(&dir)).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn pinned_urls() {
        assert_eq!(
            LAYOUT.url(),
            "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/resolve/46bbdf188bb0a772c08aed74882ce7e51a8f1ea6/inference.onnx"
        );
    }
}
