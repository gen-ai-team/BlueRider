# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi>=0.110",
#     "uvicorn[standard]>=0.27",
#     "python-dotenv>=1.0",
# ]
# ///
"""
Exhibition web interface for the autonomous-art driver.

Reads the same OUTPUT_DIR that run.py writes to and serves:

  GET  /                      -> the exhibition page
  GET  /api/state             -> full snapshot (series + iterations + path + status)
  GET  /output/art_NNN.png    -> rendered images
  GET  /static/*              -> frontend assets

Run:
  uv run web.py                          # 0.0.0.0:8765
  uv run web.py --host 127.0.0.1 --port 9000
  uv run web.py --output /path/to/output

The page auto-refreshes its data every few seconds, so it picks up new
iterations as run.py writes them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import deque
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path(os.getenv("OUTPUT_DIR", REPO_ROOT / "output")).resolve()
STATIC_DIR = REPO_ROOT / "web"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Exhibition viewer for image_gen.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help="Path to the run.py OUTPUT_DIR (default: ./output).")
    ap.add_argument("--reload", action="store_true",
                    help="Enable uvicorn auto-reload (dev).")
    return ap.parse_args()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _tail_log(path: Path, n: int = 12) -> list[str]:
    """Return the last n non-empty log lines."""
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            # read last ~64kb — plenty for tail
            try:
                f.seek(-65536, os.SEEK_END)
            except OSError:
                f.seek(0)
            data = f.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    lines = [ln for ln in lines if ln.strip()]
    return list(deque(lines, maxlen=n))


def build_app(output_dir: Path) -> FastAPI:
    app = FastAPI(title="image_gen exhibition", docs_url=None, redoc_url=None)

    state_dir = output_dir / "state"
    series_json = state_dir / "series.json"
    path_json = state_dir / "path.json"
    iter_dir = state_dir / "iterations"
    log_path = output_dir / "run.log"

    @app.get("/api/state")
    def api_state() -> JSONResponse:
        if not output_dir.exists():
            raise HTTPException(500, f"OUTPUT_DIR not found: {output_dir}")

        series_blob = _read_json(series_json) or {"series": []}
        path_obj = _read_json(path_json)

        iterations: dict[int, dict] = {}
        if iter_dir.exists():
            for p in sorted(iter_dir.glob("iter_*.json")):
                m = re.match(r"iter_(\d+)\.json$", p.name)
                if not m:
                    continue
                blob = _read_json(p)
                if blob is None:
                    continue
                iterations[int(m.group(1))] = blob

        # Newest-first ordering of iterations; also expose the newest one
        # explicitly as "current" (= latest completed image).
        ordered = [iterations[k] for k in sorted(iterations.keys(), reverse=True)]
        current = ordered[0] if ordered else None

        # Series ordered newest -> oldest, with open series first if present.
        all_series = list(series_blob.get("series", []))
        open_series = [s for s in all_series if s.get("closed_at") is None]
        closed_series = [s for s in all_series if s.get("closed_at") is not None]
        closed_series.sort(key=lambda s: s.get("closed_at") or 0, reverse=True)
        series_sorted = open_series + closed_series

        tail = _tail_log(log_path, n=14)

        return JSONResponse({
            "output_dir": str(output_dir),
            "current": current,
            "iterations": ordered,          # newest -> oldest
            "series": series_sorted,        # open first, then newest-closed
            "path": path_obj,
            "log_tail": tail,
            "counts": {
                "iterations": len(iterations),
                "series": len(all_series),
                "open_series": len(open_series),
            },
        })

    @app.get("/output/{name}")
    def output_file(name: str):
        # Only allow images directly under OUTPUT_DIR, no traversal.
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(400, "bad name")
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+\.(png|jpe?g|webp)", name):
            raise HTTPException(404, "not an image")
        fp = output_dir / name
        if not fp.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(fp, headers={"Cache-Control": "public, max-age=60"})

    @app.get("/")
    def index():
        idx = STATIC_DIR / "index.html"
        if not idx.exists():
            raise HTTPException(500, f"frontend missing: {idx}")
        return FileResponse(idx)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def main() -> None:
    args = _parse_args()
    output_dir = args.output.resolve()
    os.environ["IMAGE_GEN_OUTPUT"] = str(output_dir)

    import uvicorn

    if args.reload:
        # reload mode needs an import string
        uvicorn.run("web:app_factory", host=args.host, port=args.port,
                    reload=True, factory=True)
    else:
        app = build_app(output_dir)
        print(f"serving {output_dir} on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def app_factory() -> FastAPI:
    """For `uvicorn --reload` (needs an import string)."""
    output_dir = Path(os.environ.get("IMAGE_GEN_OUTPUT", DEFAULT_OUTPUT)).resolve()
    return build_app(output_dir)


if __name__ == "__main__":
    main()
