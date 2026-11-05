//! Runs llama-server as a child process for the general model and stops it when dropped, so a
//! crash or Ctrl-C in the app doesn't leave it holding RAM or GPU memory.
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, thiserror::Error)]
pub enum SidecarError {
    #[error("llama-server not found; set MAKAAR_LLAMA_SERVER or put llama-server on PATH")]
    NotFound,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("llama-server exited during start-up ({0}); see {1}")]
    Exited(String, PathBuf),
    #[error("llama-server not ready after {0:?}; see {1}")]
    Timeout(Duration, PathBuf),
}

pub struct SidecarConfig {
    pub model: PathBuf,
    pub mmproj: PathBuf,
    pub threads: usize,
    /// Layers on the GPU: `Some(0)` forces CPU; `None` leaves llama-server's default.
    pub gpu_layers: Option<u32>,
    pub ctx: u32,
    /// llama-server's output goes here, for diagnosing start-up failures.
    pub log: PathBuf,
}

pub struct LlamaServer {
    child: Child,
    pub url: String,
}

/// `$MAKAAR_LLAMA_SERVER`, else `llama-server` beside this executable, else on PATH.
pub fn find_binary() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("MAKAAR_LLAMA_SERVER") {
        return Some(p.into());
    }
    let name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let beside = std::env::current_exe().ok()?.parent()?.join(name);
    if beside.is_file() {
        return Some(beside);
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join(name))
            .find(|p| p.is_file())
    })
}

fn free_port() -> std::io::Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?
        .local_addr()?
        .port())
}

impl LlamaServer {
    pub fn start(bin: &Path, cfg: &SidecarConfig) -> Result<Self, SidecarError> {
        let port = free_port()?;
        let log = std::fs::File::create(&cfg.log)?;
        let mut cmd = Command::new(bin);
        cmd.arg("-m")
            .arg(&cfg.model)
            .arg("--mmproj")
            .arg(&cfg.mmproj)
            .args(["-c", &cfg.ctx.to_string(), "-t", &cfg.threads.to_string()])
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--no-webui",
            ])
            .stdin(Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log);
        if let Some(n) = cfg.gpu_layers {
            cmd.args(["-ngl", &n.to_string()]);
            if n == 0 {
                // a CUDA build still reserves GPU memory with 0 layers unless it can't see the GPU
                cmd.env("CUDA_VISIBLE_DEVICES", "");
            }
        }
        let mut server = Self {
            child: cmd.spawn()?,
            url: format!("http://127.0.0.1:{port}"),
        };
        server.wait_ready(Duration::from_secs(300), &cfg.log)?;
        Ok(server)
    }

    fn wait_ready(&mut self, limit: Duration, log: &Path) -> Result<(), SidecarError> {
        let t = Instant::now();
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(2)))
            .http_status_as_error(false)
            .build()
            .new_agent();
        while t.elapsed() < limit {
            if let Some(status) = self.child.try_wait()? {
                return Err(SidecarError::Exited(status.to_string(), log.into()));
            }
            let ok = agent
                .get(&format!("{}/health", self.url))
                .call()
                .is_ok_and(|r| r.status() == 200);
            if ok {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        Err(SidecarError::Timeout(limit, log.into()))
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The general model behind llama-server, started on the first region that needs it. Pages the
/// specialists finish never load the 1.8 GB model.
pub struct LazyServer {
    bin: PathBuf,
    cfg: SidecarConfig,
    running: Option<(LlamaServer, crate::engine::OpenAiEngine)>,
    name: String,
}

impl LazyServer {
    pub fn new(name: &str, bin: PathBuf, cfg: SidecarConfig) -> Self {
        Self {
            bin,
            cfg,
            running: None,
            name: name.into(),
        }
    }

    pub fn started(&self) -> bool {
        self.running.is_some()
    }
}

impl crate::engine::Engine for LazyServer {
    fn name(&self) -> &str {
        &self.name
    }

    fn recognize(
        &mut self,
        crop: &image::RgbImage,
        task: crate::engine::Task,
        max_tokens: u32,
    ) -> Result<crate::engine::Recognition, crate::engine::EngineError> {
        if self.running.is_none() {
            let t = Instant::now();
            let server = LlamaServer::start(&self.bin, &self.cfg)
                .map_err(|e| crate::engine::EngineError::Response(e.to_string()))?;
            eprintln!("general model ready in {:.0}s", t.elapsed().as_secs_f64());
            let engine = crate::engine::OpenAiEngine::new(&self.name, &server.url);
            self.running = Some((server, engine));
        }
        let (_, engine) = self.running.as_mut().expect("started above");
        engine.recognize(crop, task, max_tokens)
    }
}
