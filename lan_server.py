import argparse
import html
import json
import mimetypes
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
GEN_SCRIPT = ROOT / "gen.py"
DEFAULT_CHECKPOINT = ROOT / "checkpoints_256_clean" / "latest.pt"
OUTPUT_DIR = ROOT / "cgan-images-256-clean" / "lan"


def clamp_int(value: str, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def validate_float_list(value: str, name: str) -> str:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise ValueError(f"{name} cannot be empty.")
    if len(parts) > 64:
        raise ValueError(f"{name} has too many values.")
    for part in parts:
        try:
            parsed = float(part)
        except ValueError as exc:
            raise ValueError(f"{name} must contain comma-separated numbers.") from exc
        if parsed < 0.0 or parsed > 1.0:
            raise ValueError(f"{name} values must be between 0 and 1.")
    return ",".join(parts)


def first(params: dict[str, list[str]], key: str, default: str) -> str:
    values = params.get(key)
    if not values:
        return default
    return values[0]


def build_generation_command(params: dict[str, list[str]]) -> tuple[list[str], Path]:
    digit = clamp_int(first(params, "digit", "7"), "digit", 0, 9)
    num = clamp_int(first(params, "num", "16"), "num", 1, 64)
    nrow = clamp_int(first(params, "nrow", "8"), "nrow", 1, 16)
    seed = clamp_int(first(params, "seed", "123"), "seed", 0, 2_147_483_647)
    threshold = clamp_int(first(params, "clean_threshold", "210"), "clean_threshold", 0, 255)
    thickness = validate_float_list(first(params, "thickness", "0.5"), "thickness")
    slant = validate_float_list(first(params, "slant", "0.5"), "slant")

    device = first(params, "device", "cuda").lower()
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu.")

    checkpoint = Path(first(params, "checkpoint", str(DEFAULT_CHECKPOINT)))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"digit{digit}_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"

    command = [
        sys.executable,
        str(GEN_SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(output_path),
        "--digit",
        str(digit),
        "--num",
        str(num),
        "--nrow",
        str(nrow),
        "--thickness",
        thickness,
        "--slant",
        slant,
        "--seed",
        str(seed),
        "--device",
        device,
        "--clean-threshold",
        str(threshold),
    ]

    if first(params, "clean", "0").lower() in {"1", "true", "yes", "on"}:
        command.append("--clean")

    return command, output_path


class Handler(BaseHTTPRequestHandler):
    server_version = "CGANLanServer/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_index()
            return
        if parsed.path == "/generate":
            self.generate(parse_qs(parsed.query), as_json=False)
            return
        if parsed.path == "/api/generate":
            self.generate(parse_qs(parsed.query), as_json=True)
            return
        if parsed.path.startswith("/images/"):
            self.send_image(parsed.path.removeprefix("/images/"))
            return
        self.send_error(404, "Not found")

    def generate(self, params: dict[str, list[str]], as_json: bool) -> None:
        try:
            command, output_path = build_generation_command(params)
            result = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode != 0:
                raise RuntimeError((result.stderr or result.stdout or "generation failed").strip())
            if not output_path.exists():
                raise RuntimeError("generation finished but output image was not created.")
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)
            return

        if as_json:
            self.send_json(
                {
                    "ok": True,
                    "url": f"/images/{output_path.name}",
                    "path": str(output_path),
                    "message": result.stdout.strip(),
                }
            )
            return

        data = output_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_index(self) -> None:
        body = """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CGAN digit generator</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:760px;margin:32px auto;padding:0 16px;line-height:1.5}
form{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}
label{display:grid;gap:4px}
input,select,button{font:inherit;padding:8px}
button{grid-column:1/-1;cursor:pointer}
img{max-width:100%;border:1px solid #ddd}
</style>
<h1>CGAN digit generator</h1>
<form action="/generate" method="get">
  <label>digit <input name="digit" type="number" min="0" max="9" value="7"></label>
  <label>num <input name="num" type="number" min="1" max="64" value="16"></label>
  <label>nrow <input name="nrow" type="number" min="1" max="16" value="8"></label>
  <label>thickness <input name="thickness" value="0.5"></label>
  <label>slant <input name="slant" value="0.5"></label>
  <label>seed <input name="seed" type="number" min="0" value="123"></label>
  <label>device <select name="device"><option>cuda</option><option>cpu</option></select></label>
  <label>clean <select name="clean"><option value="0">off</option><option value="1">on</option></select></label>
  <button type="submit">Generate PNG</button>
</form>
</html>"""
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_image(self, filename: str) -> None:
        safe_name = Path(unquote(filename)).name
        path = (OUTPUT_DIR / safe_name).resolve()
        if path.parent != OUTPUT_DIR.resolve() or not path.exists():
            self.send_error(404, "Image not found")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.address_string(), self.log_date_time_string(), html.escape(format % args))
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expose gen.py on the LAN over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}")
    print("Open http://<this-computer-lan-ip>:%d from another device." % args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
