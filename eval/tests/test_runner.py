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
