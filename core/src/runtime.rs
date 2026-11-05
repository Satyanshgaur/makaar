//! The llama.cpp runtime for the general model: `makaar setup` downloads llama.cpp's official
//! prebuilt release for this platform, pinned to one build and checked against the SHA-256
//! GitHub publishes for it, and unpacks it under the models directory.
use std::path::{Path, PathBuf};

use crate::models::{fetch_verified, ModelError};

/// llama.cpp build all assets below come from (MIT). PaddleOCR-VL needs b8110 or later.
pub const LLAMA_TAG: &str = "b12137";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlamaAsset {
    pub name: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

const fn asset(name: &'static str, sha256: &'static str, size: u64) -> LlamaAsset {
    LlamaAsset { name, sha256, size }
}

/// GPU backend of the downloaded build. `Cpu` runs everywhere; Vulkan uses most laptop GPUs
/// (Intel, AMD, NVIDIA) but needs a Vulkan driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Vulkan,
}

/// The release asset for this OS/architecture and backend, if llama.cpp publishes one.
pub fn asset_for(os: &str, arch: &str, backend: Backend) -> Option<LlamaAsset> {
    Some(match (os, arch, backend) {
        ("linux", "x86_64", Backend::Cpu) => asset(
            "llama-b12137-bin-ubuntu-x64.tar.gz",
            "d9be8922022f26524efe5af7449b50414f0fc781e9b740042e4af7e50e511525",
            23_077_357,
        ),
        ("linux", "x86_64", Backend::Vulkan) => asset(
            "llama-b12137-bin-ubuntu-vulkan-x64.tar.gz",
            "e008a270589da89571c963643d8a8072835e34120e4cc68cb052c7620ec5b068",
            35_887_314,
        ),
        ("linux", "aarch64", Backend::Cpu) => asset(
            "llama-b12137-bin-ubuntu-arm64.tar.gz",
            "19bfb23aba0d52d0e32b22e8c13f776005100e6c606c858293df9a17ab4591e7",
            18_753_296,
        ),
        ("linux", "aarch64", Backend::Vulkan) => asset(
            "llama-b12137-bin-ubuntu-vulkan-arm64.tar.gz",
            "343e3301cca77832718b766c3e25efc21291f5326b99d938f165183ab2134da9",
            29_198_034,
        ),
        // macOS builds use Metal on Apple silicon either way
        ("macos", "aarch64", _) => asset(
            "llama-b12137-bin-macos-arm64.tar.gz",
            "65fc3b01ce7def2177982759b0af8c67221986114ef4c0a56e1e6b9218fa2185",
            16_670_227,
        ),
        ("macos", "x86_64", _) => asset(
            "llama-b12137-bin-macos-x64.tar.gz",
            "bbd4073ae799354e72b009b43adb3b960ac72a0c737ed0043924fb199ac5959c",
            12_408_230,
        ),
        ("windows", "x86_64", Backend::Cpu) => asset(
            "llama-b12137-bin-win-cpu-x64.zip",
            "20019b4103c0da8d1b7b65c4a1e542eba4e081ff2d584a7b8ab4723afc835b8d",
            24_728_320,
        ),
        ("windows", "x86_64", Backend::Vulkan) => asset(
            "llama-b12137-bin-win-vulkan-x64.zip",
            "6c30af8aa3ac882cd409a13f912d2f69858d05e8f9d7a4ff6d7e361d487e02b1",
            37_677_976,
        ),
        ("windows", "aarch64", Backend::Cpu) => asset(
            "llama-b12137-bin-win-cpu-arm64.zip",
            "e33cbd04ed698b8b56e44d6db80f305eace0f4861d3bccef0694c73f18bdfe37",
            13_222_600,
        ),
        ("windows", "aarch64", Backend::Vulkan) => asset(
            "llama-b12137-bin-win-vulkan-arm64.zip",
            "5d135f44169e3c1c79fbcc5da98b0c0367da5d400f8ca65d9f4eefff5d5c1bc6",
            25_752_329,
        ),
        _ => return None,
    })
}

pub fn current_asset(backend: Backend) -> Option<LlamaAsset> {
    asset_for(std::env::consts::OS, std::env::consts::ARCH, backend)
}

/// Where a setup unpacks llama.cpp: `<models>/llama.cpp/<tag>/`.
pub fn install_dir(models: &Path) -> PathBuf {
    models.join("llama.cpp").join(LLAMA_TAG)
}

fn server_name() -> &'static str {
    if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

/// The installed llama-server, searched under `install_dir` (archives nest it differently).
pub fn installed_server(models: &Path) -> Option<PathBuf> {
    find_file(&install_dir(models), server_name(), 3)
}

fn find_file(dir: &Path, name: &str, depth: u32) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut dirs = vec![];
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            dirs.push(p);
        } else if p.file_name().is_some_and(|n| n == name) {
            return Some(p);
        }
    }
    if depth == 0 {
        return None;
    }
    dirs.into_iter()
        .find_map(|d| find_file(&d, name, depth - 1))
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("zip: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("llama.cpp publishes no {1:?} build for {0}")]
    Unsupported(String, Backend),
    #[error("archive has no {0}")]
    NoServer(&'static str),
}

/// Downloads, verifies and unpacks the build; returns the llama-server path.
pub fn install(
    models: &Path,
    backend: Backend,
    progress: impl FnMut(u64, u64),
) -> Result<PathBuf, RuntimeError> {
    let a = current_asset(backend).ok_or_else(|| {
        RuntimeError::Unsupported(
            format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            backend,
        )
    })?;
    let dir = install_dir(models);
    let archive = models.join("llama.cpp").join(a.name);
    let url = format!(
        "https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{}",
        a.name
    );
    fetch_verified(&url, &archive, a.sha256, a.size, progress)?;
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    unpack(&archive, &dir)?;
    std::fs::remove_file(&archive)?;
    installed_server(models).ok_or(RuntimeError::NoServer(server_name()))
}

/// Unpacks a `.tar.gz` or `.zip` into `dir`, refusing entries that would escape it.
pub fn unpack(archive: &Path, dir: &Path) -> Result<(), RuntimeError> {
    std::fs::create_dir_all(dir)?;
    let file = std::fs::File::open(archive)?;
    if archive.extension().is_some_and(|e| e == "zip") {
        // enclosed_name() rejects absolute paths and `..`
        zip::ZipArchive::new(file)?.extract(dir)?;
    } else {
        let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
        // unpack_in() skips entries with `..` or absolute paths; preserves the executable bit
        for entry in tar.entries()? {
            entry?.unpack_in(dir)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_desktop_platform_has_a_cpu_build() {
        for (os, arch) in [
            ("linux", "x86_64"),
            ("linux", "aarch64"),
            ("macos", "aarch64"),
            ("macos", "x86_64"),
            ("windows", "x86_64"),
            ("windows", "aarch64"),
        ] {
            let a = asset_for(os, arch, Backend::Cpu).unwrap();
            assert!(
                a.name.contains(LLAMA_TAG) && a.sha256.len() == 64,
                "{os}-{arch}"
            );
        }
        assert!(asset_for("freebsd", "x86_64", Backend::Cpu).is_none());
    }

    #[test]
    fn unpacks_tar_gz_and_finds_server() {
        let models = std::env::temp_dir().join(format!("makaar-rt-{}", std::process::id()));
        let src = models.join("src");
        std::fs::create_dir_all(src.join(format!("llama-{LLAMA_TAG}"))).unwrap();
        std::fs::write(
            src.join(format!("llama-{LLAMA_TAG}")).join(server_name()),
            b"bin",
        )
        .unwrap();
        let archive = models.join("a.tar.gz");
        let gz = flate2::write::GzEncoder::new(
            std::fs::File::create(&archive).unwrap(),
            flate2::Compression::fast(),
        );
        let mut tar = tar::Builder::new(gz);
        tar.append_dir_all(".", &src).unwrap();
        tar.into_inner().unwrap().finish().unwrap();

        unpack(&archive, &install_dir(&models)).unwrap();
        let server = installed_server(&models).unwrap();
        assert!(server.ends_with(Path::new(&format!("llama-{LLAMA_TAG}")).join(server_name())));
        let _ = std::fs::remove_dir_all(models);
    }
}
