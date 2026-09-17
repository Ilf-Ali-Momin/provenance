"""Local annotation server for the 200 passage set.

DATA.md requires the annotation set to be built by hand, and D035 says nothing
in this project may supply the gold standard. This tool does not supply one: it
serves the passages, records the spans a human marks, and writes them back. No
model is called, and the extractor's predictions are deliberately unreachable
from here - the guideline's Procedure 4 says not to look at extractor output
while annotating, and the way to keep that rule is to not serve it.

    PYTHONPATH=. .venv/bin/python -m tools.annotate

Then open http://127.0.0.1:8777/. Writes land in eval/annotation/passages.jsonl
as they are made, atomically, so a closed laptop costs nothing.

Bound to 127.0.0.1 with no auth, which is correct for a tool that writes to the
working tree and wrong for anything reachable. stdlib only - this is a local
scratch tool and does not justify a dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

ANNOTATIONS: Final = Path("eval/annotation/passages.jsonl")
PAGE: Final = Path(__file__).with_name("annotate.html")
TYPES: Final = frozenset({"factual_assertion", "opinion", "prediction", "attribution"})
MAX_BODY: Final = 8 * 1024 * 1024

_lock = threading.Lock()


def load(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace the file atomically, so an interrupted write cannot truncate it."""
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(body)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def validate(row: dict[str, Any], text: str) -> list[dict[str, Any]]:
    """Reject spans the scorer could not use, rather than storing them.

    The scorer reads `start`, `end` and `type` and slices the passage with them.
    A span outside the text, inverted, or carrying a type outside the four would
    be a gold annotation nothing can score, discovered at scoring time.
    """
    clean: list[dict[str, Any]] = []
    for span in row.get("spans") or []:
        start, end = int(span["start"]), int(span["end"])
        kind = str(span["type"])
        if kind not in TYPES:
            raise ValueError(f"unknown type {kind!r}")
        if not 0 <= start < end <= len(text):
            raise ValueError(f"span {start}-{end} outside a passage of {len(text)} characters")
        clean.append({"start": start, "end": end, "type": kind})
    clean.sort(key=lambda s: s["start"])
    for a, b in pairwise(clean):
        if a["end"] > b["start"]:
            msg = f"overlapping spans {a['start']}-{a['end']} and {b['start']}-{b['end']}"
            raise ValueError(msg)
    return clean


def handler_for(path: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            route = self.path.split("?")[0]
            if route in {"/", "/index.html"}:
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif route == "/passages":
                with _lock:
                    rows = load(path)
                self._send(200, json.dumps(rows).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path.split("?")[0] != "/save":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._send(413, b"too large", "text/plain")
                return
            try:
                incoming = json.loads(self.rfile.read(length) or b"[]")
            except ValueError:
                self._send(400, b"bad json", "text/plain")
                return

            with _lock:
                rows = load(path)
                index = {str(r["passage_id"]): n for n, r in enumerate(rows)}
                try:
                    for row in incoming:
                        at = index.get(str(row.get("passage_id")))
                        if at is None:
                            raise ValueError(f"unknown passage {row.get('passage_id')!r}")
                        # The text is never taken from the client. Offsets index
                        # the stored passage, so the stored passage is the one
                        # they are checked against.
                        target = rows[at]
                        target["spans"] = validate(row, str(target["text"]))
                        target["notes"] = str(row.get("notes") or "")
                        target["reviewed"] = bool(row.get("reviewed"))
                except (ValueError, KeyError, TypeError) as exc:
                    self._send(400, str(exc).encode(), "text/plain")
                    return
                write(path, rows)
                done = sum(1 for r in rows if r.get("reviewed"))
            self._send(200, json.dumps({"reviewed": done}).encode(), "application/json")

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    if not args.annotations.exists():
        raise SystemExit(f"no annotation file at {args.annotations}; run eval.annotation.sample")

    rows = load(args.annotations)
    done = sum(1 for r in rows if r.get("reviewed"))
    url = f"http://127.0.0.1:{args.port}/"
    sys.stdout.write(
        f"{len(rows)} passages, {done} reviewed\n"
        f"guideline: docs/annotation-guideline.md\n"
        f"serving {url}  (ctrl-c to stop; every span is saved as you mark it)\n"
    )
    if not args.no_open:
        webbrowser.open(url)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(args.annotations))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        rows = load(args.annotations)
        done = sum(1 for r in rows if r.get("reviewed"))
        sys.stdout.write(f"\nstopped. {done} of {len(rows)} reviewed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
