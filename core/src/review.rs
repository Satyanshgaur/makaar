//! Review screen: a local web page for checking and fixing a converted document. Shows each page
//! scan with its block boxes (flagged red, verified green) beside the block texts, which can be
//! edited with a live preview (math as MathML). Saving writes the document JSON and calls back so
//! the caller can re-export. Listens on 127.0.0.1 only, and every API call needs the session
//! token from the printed URL, so other local pages can't read or overwrite the notes.
use std::path::PathBuf;

use tiny_http::{Header, Method, Request, Response, Server};

use crate::document::{BlockType, Document};
use crate::export::strip_math_delims;
use crate::latex::{math_segments, Seg};
use crate::omml::mathml;

const PAGE: &str = include_str!("review.html");

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// HTML preview of a block: prose escaped (line breaks kept), math as MathML; math that doesn't
/// parse is shown as red source so the reviewer sees what will fall back.
pub fn preview(kind: BlockType, label: Option<&str>, text: &str) -> String {
    let math = |m: &str, display: bool| match mathml(m.trim(), display) {
        Ok(x) => x,
        Err(e) => format!(
            "<code class=\"bad\" title=\"{}\">{}</code>",
            esc(&e.0),
            esc(m)
        ),
    };
    if kind == BlockType::Formula {
        return math(strip_math_delims(text), label != Some("inline_formula"));
    }
    if label == Some("algorithm") && !text.contains("\\(") {
        return format!("<pre>{}</pre>", esc(text));
    }
    math_segments(text)
        .into_iter()
        .map(|s| match s {
            Seg::Text(t) => esc(t).replace('\n', "<br>"),
            Seg::Inline(m) => math(m, false),
            Seg::Display(m) => math(m, true),
        })
        .collect()
}

fn token() -> String {
    // 128 bits from the OS's hasher seeds plus time; enough to stop guessing from other pages.
    use std::hash::{BuildHasher, Hasher};
    let mut out = String::new();
    for i in 0..2u64 {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(i);
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        );
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}

/// A loopback server on `port` (0: any free one) and its page URL with a fresh session token.
fn bind(port: u16) -> std::io::Result<(Server, u16, String)> {
    let server = Server::http(("127.0.0.1", port)).map_err(std::io::Error::other)?;
    let port = server.server_addr().to_ip().map_or(port, |a| a.port());
    Ok((
        server,
        port,
        format!("http://127.0.0.1:{port}/?t={}", token()),
    ))
}

pub struct Review {
    pub doc: Document,
    pub json_path: PathBuf,
}

type Saver<'a> = dyn FnMut(&mut Document) -> Result<String, String> + 'a;
/// A save callback that can move to the server thread.
pub type SendSaver = Box<dyn FnMut(&mut Document) -> Result<String, String> + Send>;

impl Review {
    /// Serves until the process is stopped, opening the browser. `on_save` re-exports and
    /// returns a status line.
    pub fn serve(self, port: u16, on_save: &mut Saver) -> std::io::Result<()> {
        let (server, port, url) = bind(port)?;
        eprintln!("review at {url}  (Ctrl-C to stop)");
        open_browser(&url);
        self.run(server, port, &url, on_save);
        Ok(())
    }

    /// Serves on a background thread (the desktop app) and returns the page URL at once.
    pub fn spawn(self, on_save: SendSaver) -> std::io::Result<String> {
        let (server, port, url) = bind(0)?;
        let page = url.clone();
        std::thread::spawn(move || {
            let mut on_save = on_save;
            self.run(server, port, &page, on_save.as_mut());
        });
        Ok(url)
    }

    fn run(mut self, server: Server, port: u16, url: &str, on_save: &mut Saver) {
        let token = url.rsplit_once("t=").map_or("", |(_, t)| t).to_string();
        for req in server.incoming_requests() {
            let _ = self.handle(req, port, &token, on_save);
        }
    }

    /// The document and every block's preview, inlined into the page so it renders at once
    /// instead of fetching a preview per block.
    fn boot(&self) -> std::io::Result<String> {
        let mut previews = serde_json::Map::new();
        for (i, p) in self.doc.pages.iter().enumerate() {
            for b in &p.blocks {
                let html = preview(
                    b.kind.clone(),
                    b.label.as_deref(),
                    b.text.as_deref().unwrap_or(""),
                );
                previews.insert(format!("{i}:{}", b.id), html.into());
            }
        }
        let boot = serde_json::json!({ "doc": self.doc, "previews": previews });
        // `</script>` inside a string would end the script element
        Ok(format!(
            "window.BOOT = {};",
            boot.to_string().replace("</", "<\\/")
        ))
    }

    fn handle(
        &mut self,
        mut req: Request,
        port: u16,
        token: &str,
        on_save: &mut Saver,
    ) -> std::io::Result<()> {
        // DNS rebinding: only answer requests addressed to this loopback port
        let host_ok = req.headers().iter().any(|h| {
            h.field.equiv("Host")
                && [format!("127.0.0.1:{port}"), format!("localhost:{port}")]
                    .contains(&h.value.to_string())
        });
        let url = req.url().to_string();
        let (path, query) = url.split_once('?').unwrap_or((&url, ""));
        let authed = query.split('&').any(|kv| kv == format!("t={token}"));
        if !host_ok || !authed {
            return req.respond(Response::from_string("forbidden").with_status_code(403));
        }
        let ctype = |v: &str| Header::from_bytes("Content-Type", v).expect("static header");
        match (req.method().clone(), path) {
            (Method::Get, "/") => {
                let page = PAGE.replace("/*BOOT*/", &self.boot()?);
                req.respond(
                    Response::from_string(page).with_header(ctype("text/html; charset=utf-8")),
                )
            }
            (Method::Get, "/doc") => {
                let body = serde_json::to_string(&self.doc).map_err(std::io::Error::other)?;
                req.respond(Response::from_string(body).with_header(ctype("application/json")))
            }
            (Method::Get, p) if p.starts_with("/page/") => {
                let img = p["/page/".len()..]
                    .parse::<usize>()
                    .ok()
                    .and_then(|i| self.doc.pages.get(i))
                    .and_then(|pg| pg.image.clone())
                    .and_then(|path| std::fs::read(path).ok());
                match img {
                    Some(bytes) => {
                        let mime = match image::guess_format(&bytes) {
                            Ok(f) => f.to_mime_type(),
                            Err(_) => "application/octet-stream",
                        };
                        req.respond(Response::from_data(bytes).with_header(ctype(mime)))
                    }
                    None => req.respond(Response::from_string("no image").with_status_code(404)),
                }
            }
            (Method::Post, "/preview") => {
                let mut body = String::new();
                req.as_reader().read_to_string(&mut body)?;
                let v: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
                let kind: BlockType =
                    serde_json::from_value(v["type"].clone()).unwrap_or(BlockType::Text);
                let html = preview(kind, v["label"].as_str(), v["text"].as_str().unwrap_or(""));
                req.respond(
                    Response::from_string(html).with_header(ctype("text/html; charset=utf-8")),
                )
            }
            (Method::Post, "/save") => {
                let mut body = String::new();
                req.as_reader().read_to_string(&mut body)?;
                let msg = match serde_json::from_str::<Document>(&body) {
                    Ok(mut doc) => {
                        let status = on_save(&mut doc);
                        // write after the exporters, so flags they set are kept
                        let written = serde_json::to_string_pretty(&doc)
                            .map_err(|e| e.to_string())
                            .and_then(|j| {
                                std::fs::write(&self.json_path, j).map_err(|e| e.to_string())
                            });
                        self.doc = doc;
                        match (status, written) {
                            (Ok(s), Ok(())) => Ok(s),
                            (Err(e), _) | (_, Err(e)) => Err(e),
                        }
                    }
                    Err(e) => Err(format!("bad document: {e}")),
                };
                let (code, text) = match msg {
                    Ok(s) => (200, s),
                    Err(e) => (500, e),
                };
                req.respond(Response::from_string(text).with_status_code(code))
            }
            _ => req.respond(Response::from_string("not found").with_status_code(404)),
        }
    }
}

fn open_browser(url: &str) {
    let cmd = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(windows) {
        "explorer"
    } else {
        "xdg-open"
    };
    if std::env::var_os("MAKAAR_NO_BROWSER").is_none() {
        let _ = std::process::Command::new(cmd)
            .arg(url)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn previews() {
        let p = preview(BlockType::Text, Some("text"), "cost <b> is \\(c_1\\)\nnext");
        assert!(p.starts_with("cost &lt;b&gt; is <math><mrow><msub>"), "{p}");
        assert!(p.ends_with("<br>next"));
        assert!(
            preview(BlockType::Formula, Some("display_formula"), "\\[x\\]")
                .starts_with("<math display=\"block\">")
        );
        assert!(preview(BlockType::Text, None, "\\(\\frac{a}\\)").contains("class=\"bad\""));
        assert_eq!(
            preview(BlockType::Text, Some("algorithm"), "a<b"),
            "<pre>a&lt;b</pre>"
        );
    }

    #[test]
    fn tokens_differ() {
        let (a, b) = (token(), token());
        assert_eq!(a.len(), 32);
        assert_ne!(a, b);
    }
}
