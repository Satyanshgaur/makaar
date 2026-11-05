import http.server
import json
import threading
from pathlib import Path

from makaar_eval.runner import main


class Fake(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"choices": [{"message": {"content": "hello world"},
                                        "logprobs": {"content": [{"logprob": -0.1}]}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_runner_end_to_end(tmp_path: Path, capsys):
    srv = http.server.HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    d = tmp_path / "data" / "printed"
    d.mkdir(parents=True)
    (d / "p1.png").write_bytes(b"\x89PNG")
    (d / "p1.md").write_text("hello world")
    main([str(tmp_path / "data"), "--name", "fake", "--url", f"http://127.0.0.1:{srv.server_port}",
          "--out", str(tmp_path / "out")])
    res = json.loads((tmp_path / "out" / "fake.json").read_text())
    assert res["summary"]["printed"]["ned_mean"] == 0.0
    srv.shutdown()


def test_predictions_mode(tmp_path: Path):
    d = tmp_path / "data" / "printed"
    d.mkdir(parents=True)
    (d / "p1.png").write_bytes(b"\x89PNG")
    (d / "p1.md").write_text("hello world")
    pr = tmp_path / "pred" / "printed"
    pr.mkdir(parents=True)
    (pr / "p1.md").write_text("hello world")
    main([str(tmp_path / "data"), "--name", "ext", "--predictions", str(tmp_path / "pred"),
          "--out", str(tmp_path / "out")])
    assert json.loads((tmp_path / "out" / "ext.json").read_text())["summary"]["printed"]["ned_mean"] == 0.0


class Gemini(http.server.BaseHTTPRequestHandler):
    """Rejects the first call with 429, then answers; records what it was sent."""
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Gemini.seen.append((self.path, self.headers.get("Authorization"), body))
        if len(Gemini.seen) == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0.01")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        out = json.dumps({"choices": [{"message": {"content": "hello world"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_api_key_retry_and_full_url(tmp_path: Path):
    srv = http.server.HTTPServer(("127.0.0.1", 0), Gemini)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    d = tmp_path / "data" / "printed"
    d.mkdir(parents=True)
    (d / "p1.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    (d / "p1.md").write_text("hello world")
    (tmp_path / "key").write_text("sekrit\n")
    (tmp_path / "prompt.txt").write_text("Transcribe.\n")
    url = f"http://127.0.0.1:{srv.server_port}/v1beta/openai/chat/completions"
    main([str(tmp_path / "data"), "--name", "g", "--url", url, "--model", "gemini-x",
          "--api-key-file", str(tmp_path / "key"), "--no-logprobs", "--prompt-file", str(tmp_path / "prompt.txt"),
          "--reasoning-effort", "low", "--out", str(tmp_path / "out")])
    srv.shutdown()
    path, auth, body = Gemini.seen[-1]
    assert len(Gemini.seen) == 2
    assert path == "/v1beta/openai/chat/completions"
    assert auth == "Bearer sekrit"
    assert "logprobs" not in body and body["reasoning_effort"] == "low" and body["model"] == "gemini-x"
    assert body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert body["messages"][0]["content"][1]["text"] == "Transcribe."
    assert json.loads((tmp_path / "out" / "g.json").read_text())["summary"]["printed"]["ned_mean"] == 0.0
