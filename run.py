# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.40",
#     "python-dotenv>=1.0",
#     "pillow>=10",
# ]
# ///
"""
Autonomous generative-art driver with codified series + structured outputs.

Artifacts live in OUTPUT_DIR (default ./output):

    art.py                    latest generated script
    art_NNN.png               rendered images
    state/
        series.json           manifest of all series
        path.json             current artistic_path (structured)
        iterations/
            iter_NNN.json     canonical record per iteration
    exhibition_diary.md       rendered view of iterations + series
    artistic_path.md          rendered view of path.json
    debug/                    raw LLM replies with finish reasons
    run.log                   log file

Run:
    uv run run.py                 # infinite loop
    uv run run.py --fresh         # wipe output/ first (after confirmation)

Environment: see .env.example.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
PROMPT_PATH = REPO_ROOT / "prompt.md"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", REPO_ROOT / "output")).resolve()

ART_SCRIPT = OUTPUT_DIR / "art.py"
DIARY_PATH = OUTPUT_DIR / "exhibition_diary.md"
PATH_MD = OUTPUT_DIR / "artistic_path.md"
LOG_PATH = OUTPUT_DIR / "run.log"
DEBUG_DIR = OUTPUT_DIR / "debug"

STATE_DIR = OUTPUT_DIR / "state"
SERIES_JSON = STATE_DIR / "series.json"
PATH_JSON = STATE_DIR / "path.json"
ITER_DIR = STATE_DIR / "iterations"

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
VISION_MODEL = os.getenv("VISION_MODEL", MODEL)
EXEC_TIMEOUT = int(os.getenv("EXEC_TIMEOUT", "240"))
MAX_FIX_RETRIES = int(os.getenv("MAX_FIX_RETRIES", "3"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))

# series policy
MIN_SERIES_LEN = int(os.getenv("MIN_SERIES_LEN", "7"))
SOFT_MAX_SERIES_LEN = int(os.getenv("SOFT_MAX_SERIES_LEN", "15"))
HARD_MAX_SERIES_LEN = int(os.getenv("HARD_MAX_SERIES_LEN", "18"))
# rewrite artistic_path roughly every N iterations (also on every series change)
PATH_REWRITE_EVERY = 6


def _resolve_art_python() -> str:
    """Pick the interpreter used to run art.py. It must have pillow/numpy/
    scipy/noise installed. Priority:
      1. ART_PYTHON env var.
      2. .venv/bin/python next to this file.
      3. .venv/bin/python next to OUTPUT_DIR.
      4. sys.executable (warn — may be uv's ephemeral env without art deps).
    """
    explicit = os.getenv("ART_PYTHON")
    if explicit:
        return explicit
    candidates = [
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        OUTPUT_DIR / ".venv" / "bin" / "python",
        OUTPUT_DIR / ".venv" / "Scripts" / "python.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


ART_PYTHON = _resolve_art_python()


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Kandinsky autonomous-art driver.")
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Wipe OUTPUT_DIR before starting (keeps run.log handle open).",
    )
    return ap.parse_args()


ARGS = _parse_args()


def _wipe_output_dir() -> None:
    """Remove everything under OUTPUT_DIR."""
    if OUTPUT_DIR.exists():
        for child in OUTPUT_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


if ARGS.fresh:
    _wipe_output_dir()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)
ITER_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
log = logging.getLogger("art")

if not os.getenv("OPENAI_API_KEY"):
    log.error("OPENAI_API_KEY is not set. Put it in .env or export it.")
    sys.exit(1)

# Guard against leftover art_*.png from a previous (markdown-based) run when
# state/ is empty. Refuse rather than silently conflating regimes.
_pngs = list(OUTPUT_DIR.glob("art_*.png"))
_has_state = SERIES_JSON.exists()
if _pngs and not _has_state:
    log.error(
        "Found %d art_*.png files in %s but no state/ directory. "
        "Either archive them or re-run with --fresh to wipe OUTPUT_DIR.",
        len(_pngs),
        OUTPUT_DIR,
    )
    sys.exit(1)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
    timeout=LLM_TIMEOUT,
)

SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


def _check_art_python() -> None:
    """Verify the chosen interpreter can import the art dependencies."""
    probe = "import PIL, numpy, scipy, noise"
    try:
        proc = subprocess.run(
            [ART_PYTHON, "-c", probe],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.error("art interpreter %s is not runnable: %s", ART_PYTHON, e)
        sys.exit(1)
    if proc.returncode != 0:
        log.error(
            "art interpreter %s is missing required packages "
            "(pillow, numpy, scipy, noise):\n%s",
            ART_PYTHON,
            proc.stderr.strip(),
        )
        log.error(
            "Set ART_PYTHON to an interpreter that has them installed, "
            "e.g. ART_PYTHON=$(pwd)/.venv/bin/python",
        )
        sys.exit(1)
    log.info("art interpreter: %s (pillow/numpy/scipy/noise OK)", ART_PYTHON)


_check_art_python()


# --------------------------------------------------------------------------- #
# json helpers + schemas
# --------------------------------------------------------------------------- #


def _obj(
    properties: dict[str, Any],
    required: list[str] | None = None,
    additional: bool = False,
) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties.keys()),
        "additionalProperties": additional,
    }


SCHEMA_STEP3 = {
    "name": "Observation",
    "strict": True,
    "schema": _obj({"observation": {"type": "string"}}),
}

SCHEMA_STEP4 = {
    "name": "DiaryEntry",
    "strict": True,
    "schema": _obj(
        {
            "title": {"type": "string"},
            "what_i_see": {"type": "string"},
            "what_resonates": {"type": "string"},
            "what_is_silent": {"type": "string"},
            "where_next": {"type": "string"},
            "series_status": {"type": "string", "enum": ["continue", "close"]},
            "closed_reason": {"type": ["string", "null"]},
            "proposed_next_series": {
                "anyOf": [
                    {"type": "null"},
                    _obj(
                        {
                            "name": {"type": "string"},
                            "thesis": {"type": "string"},
                        }
                    ),
                ],
            },
        },
        required=[
            "title",
            "what_i_see",
            "what_resonates",
            "what_is_silent",
            "where_next",
            "series_status",
            "closed_reason",
            "proposed_next_series",
        ],
    ),
}

SCHEMA_STEP5 = {
    "name": "ArtisticPath",
    "strict": True,
    "schema": _obj(
        {
            "opening": {"type": ["string", "null"]},
            "what_attracts_me": {"type": "string"},
            "what_i_dislike": {"type": "string"},
            "what_works": {"type": "string"},
            "where_i_stumble": {"type": "string"},
            "emerging_language": {"type": "string"},
            "current_series": {"type": "string"},
            "next_hypothesis": {"type": "string"},
        },
    ),
}

SCHEMA_RETRO = {
    "name": "SeriesRetrospective",
    "strict": True,
    "schema": _obj(
        {
            "title": {
                "type": "string",
                "description": "Поэтическое имя всей серии целиком; может "
                "совпадать с series.name или уточнять его.",
            },
            "arc_summary": {
                "type": "string",
                "description": "3–5 предложений о том, как разворачивалась "
                "история серии от первой работы к последней.",
            },
            "what_emerged": {"type": "string"},
            "what_disappointed": {"type": "string"},
            "verdict": {"type": "string"},
            "bridge_to_next": {"type": "string"},
        },
    ),
}


# --------------------------------------------------------------------------- #
# state model
# --------------------------------------------------------------------------- #


@dataclass
class Series:
    id: int
    name: str
    thesis: str
    iterations: list[int] = field(default_factory=list)
    opened_at: int | None = None
    closed_at: int | None = None
    closed_reason: str | None = None
    # Set only after Шаг 4b (series retrospective) has run. Shape:
    # {title, arc_summary, what_emerged, what_disappointed, verdict,
    #  bridge_to_next}
    retrospective: dict | None = None

    def is_open(self) -> bool:
        return self.closed_at is None

    def length(self) -> int:
        return len(self.iterations)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "thesis": self.thesis,
            "iterations": list(self.iterations),
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "closed_reason": self.closed_reason,
            "retrospective": self.retrospective,
        }

    @staticmethod
    def from_dict(d: dict) -> "Series":
        return Series(
            id=d["id"],
            name=d["name"],
            thesis=d["thesis"],
            iterations=list(d.get("iterations", [])),
            opened_at=d.get("opened_at"),
            closed_at=d.get("closed_at"),
            closed_reason=d.get("closed_reason"),
            retrospective=d.get("retrospective"),
        )


class State:
    """Persistent view over state/series.json and state/iterations/*.json."""

    def __init__(self) -> None:
        self.series: list[Series] = self._load_series()
        self.iterations: dict[int, dict] = self._load_iterations()

    # -- io --
    @staticmethod
    def _load_series() -> list[Series]:
        if not SERIES_JSON.exists():
            return []
        data = json.loads(SERIES_JSON.read_text(encoding="utf-8"))
        return [Series.from_dict(s) for s in data.get("series", [])]

    @staticmethod
    def _load_iterations() -> dict[int, dict]:
        out: dict[int, dict] = {}
        for p in sorted(ITER_DIR.glob("iter_*.json")):
            m = re.match(r"iter_(\d+)\.json$", p.name)
            if not m:
                continue
            out[int(m.group(1))] = json.loads(p.read_text(encoding="utf-8"))
        return out

    def save_series(self) -> None:
        SERIES_JSON.write_text(
            json.dumps(
                {"series": [s.to_dict() for s in self.series]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def save_iteration(self, iter_json: dict) -> None:
        n = iter_json["iteration"]
        path = ITER_DIR / f"iter_{n:03d}.json"
        path.write_text(
            json.dumps(iter_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.iterations[n] = iter_json

    # -- queries --
    def next_iteration_number(self) -> int:
        return max(self.iterations) + 1 if self.iterations else 1

    def current_series(self) -> Series | None:
        for s in reversed(self.series):
            if s.is_open():
                return s
        return None

    def last_closed_series(self) -> Series | None:
        for s in reversed(self.series):
            if not s.is_open():
                return s
        return None

    def iterations_in_series(self, series_id: int) -> list[dict]:
        return [
            self.iterations[i]
            for i in next(s.iterations for s in self.series if s.id == series_id)
        ]

    # -- mutations --
    def open_series(self, name: str, thesis: str, at_iter: int) -> Series:
        new_id = (self.series[-1].id + 1) if self.series else 1
        s = Series(id=new_id, name=name, thesis=thesis, opened_at=at_iter)
        self.series.append(s)
        self.save_series()
        log.info("opened series #%d: «%s»", s.id, s.name)
        return s

    def close_series(self, at_iter: int, reason: str) -> None:
        s = self.current_series()
        if s is None:
            raise RuntimeError("no open series to close")
        s.closed_at = at_iter
        s.closed_reason = reason
        self.save_series()
        log.info("closed series #%d «%s»: %s", s.id, s.name, reason)

    def append_to_current_series(self, iter_n: int) -> None:
        s = self.current_series()
        if s is None:
            raise RuntimeError("no open series to append to")
        if iter_n not in s.iterations:
            s.iterations.append(iter_n)
            self.save_series()

    def set_series_retrospective(self, series_id: int, retro: dict) -> None:
        for s in self.series:
            if s.id == series_id:
                s.retrospective = retro
                self.save_series()
                return
        raise RuntimeError(f"series #{series_id} not found")


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #


def read_text(path: Path, default: str = "") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else default


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def encode_image(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def encode_image_downscaled(path: Path, size: int = 512, quality: int = 85) -> str:
    """Return data URL of the image resized to fit `size`x`size` and re-encoded
    as JPEG. Used for batched series retrospectives where many images are
    packed into one request."""
    import io

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def load_path_json() -> dict | None:
    if not PATH_JSON.exists():
        return None
    return json.loads(PATH_JSON.read_text(encoding="utf-8"))


def save_path_json(obj: dict) -> None:
    PATH_JSON.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# llm wrapper
# --------------------------------------------------------------------------- #


def _dump_debug(tag: str, content: str, *, finish: str | None, tokens: tuple) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", tag)
    path = DEBUG_DIR / f"{ts}_{safe}.txt"
    header = (
        f"# finish={finish} tokens_in={tokens[0]} "
        f"tokens_out={tokens[1]} chars={len(content)}\n\n"
    )
    path.write_text(header + content, encoding="utf-8")


def _llm_raw(
    user: str | list[dict],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: dict | None,
    tag: str,
) -> tuple[str, str | None]:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        # "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    content = (choice.message.content or "").strip()
    finish = getattr(choice, "finish_reason", None)
    usage = getattr(resp, "usage", None)
    u_in = getattr(usage, "prompt_tokens", "?") if usage else "?"
    u_out = getattr(usage, "completion_tokens", "?") if usage else "?"

    _dump_debug(tag, content, finish=finish, tokens=(u_in, u_out))

    level = logging.WARNING if finish == "length" else logging.INFO
    log.log(
        level,
        "    llm[%s] finish=%s tokens=in:%s/out:%s chars=%d",
        tag,
        finish,
        u_in,
        u_out,
        len(content),
    )
    return content, finish


_FENCE_RE = re.compile(
    r"^\s*```[^\S\r\n]*([A-Za-z0-9_+-]*)[^\S\r\n]*\r?\n(.*?)\r?\n```\s*\Z",
    re.DOTALL,
)


def _strip_code_fences(text: str, *, prefer_lang: str | None = None) -> str:
    """Strip a single enclosing markdown code fence, if present.

    Handles ```lang\n...\n``` and ```\n...\n```; tolerates extra leading/
    trailing whitespace. If no fence is found, returns text stripped.
    If `prefer_lang` is given and the fence language differs (e.g. ```json
    when we wanted python), the fence is still stripped — we always prefer
    the inner payload.
    """
    if not text:
        return text
    stripped = text.strip()
    m = _FENCE_RE.match(stripped)
    if m:
        return m.group(2).strip()
    # Fallback: the model sometimes emits a leading fence and then trails off
    # without closing it (or vice versa). Strip what we can find.
    if stripped.startswith("```"):
        # drop first line (the opening fence)
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
        return stripped.strip()
    return stripped


def _extract_json_blob(text: str) -> str:
    """Last-resort: pull the first {...} or [...] balanced-ish substring.
    Cheap heuristic — good enough when the model prepends/appends prose."""
    s = text.strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i = s.find(opener)
        j = s.rfind(closer)
        if i != -1 and j != -1 and j > i:
            return s[i : j + 1]
    return s


def llm_json(
    user: str | list[dict],
    schema: dict,
    *,
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    tag: str = "llm",
    max_attempts: int = 2,
) -> dict:
    """Chat completion with a strict JSON schema. Retries on parse error or
    `finish_reason == 'length'`.

    The proxy we use does not always honour `response_format=json_schema`, so
    we tolerate markdown-fenced replies and stray prose around the JSON.
    """
    rf = {"type": "json_schema", "json_schema": schema}
    last_err: str | None = None

    for attempt in range(1, max_attempts + 1):
        content, finish = _llm_raw(
            user,
            model=model or MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=rf,
            tag=f"{tag}_att{attempt}",
        )
        if finish == "length":
            last_err = f"response truncated (finish_reason=length)"
            log.warning("    %s; retrying with larger budget", last_err)
            max_tokens = int(max_tokens * 1.5) + 500
            continue

        candidates = [content, _strip_code_fences(content)]
        candidates.append(_extract_json_blob(candidates[-1]))
        parsed = None
        parse_err: Exception | None = None
        for cand in candidates:
            try:
                parsed = json.loads(cand)
                break
            except json.JSONDecodeError as e:
                parse_err = e
        if parsed is not None:
            return parsed

        last_err = f"JSON parse error: {parse_err}; body[:200]={content[:200]!r}"
        log.warning("    %s; retrying", last_err)

    raise RuntimeError(
        f"llm_json[{tag}] failed after {max_attempts} attempts: {last_err}"
    )


def llm_text(
    user: str | list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    tag: str = "llm",
    max_attempts: int = 2,
    expect_lang: str | None = None,
) -> str:
    """Chat completion that returns plain text. Strips an outer markdown
    code fence if present (``` or ```lang). Retries on empty responses or
    `finish_reason == 'length'`. Used for step 1/2 where we want raw Python
    rather than a JSON-wrapped string (escaping hurts code quality and the
    proxy does not reliably enforce response_format anyway).
    """
    last_err: str | None = None

    for attempt in range(1, max_attempts + 1):
        content, finish = _llm_raw(
            user,
            model=model or MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
            tag=f"{tag}_att{attempt}",
        )
        if finish == "length":
            last_err = "response truncated (finish_reason=length)"
            log.warning("    %s; retrying with larger budget", last_err)
            max_tokens = int(max_tokens * 1.5) + 500
            continue

        stripped = _strip_code_fences(content, prefer_lang=expect_lang)
        if stripped:
            return stripped
        last_err = "empty response"
        log.warning("    %s; retrying", last_err)

    raise RuntimeError(
        f"llm_text[{tag}] failed after {max_attempts} attempts: {last_err}"
    )


# --------------------------------------------------------------------------- #
# memory blocks (what we send to the LLM each call)
# --------------------------------------------------------------------------- #


def _path_memory_block(path_obj: dict | None) -> str:
    if not path_obj:
        return "(artistic_path ещё не создан)"
    fields_order = [
        ("opening", "Начальная мысль"),
        ("what_attracts_me", "Что меня притягивает"),
        ("what_i_dislike", "Что мне не нравится"),
        ("what_works", "Что у меня получается"),
        ("where_i_stumble", "Где я спотыкаюсь"),
        ("emerging_language", "Мой формирующийся язык"),
        ("current_series", "Текущая серия"),
        ("next_hypothesis", "Следующая гипотеза"),
    ]
    lines = []
    for key, label in fields_order:
        val = path_obj.get(key)
        if val:
            lines.append(f"- **{label}:** {val}")
    return "\n".join(lines) if lines else "(artistic_path пуст)"


def _diary_memory_block(state: State, k: int = 3) -> str:
    numbers = sorted(state.iterations.keys())[-k:]
    if not numbers:
        return "(дневник пуст)"
    blocks = []
    for n in numbers:
        it = state.iterations[n]
        blocks.append(
            f"Итерация {n:03d} — «{it['title']}»\n"
            f"(Серия #{it['series_id']} «{it['series_name']}»)\n"
            f"Что я вижу: {it['what_i_see']}\n"
            f"Что звучит: {it['what_resonates']}\n"
            f"Что молчит: {it['what_is_silent']}\n"
            f"Куда дальше: {it['where_next']}",
        )
    return "\n\n".join(blocks)


def _series_memory_block(state: State) -> str:
    cur = state.current_series()
    if cur is None:
        past = ""
        if state.series:
            past = "\nПрошлые серии:\n" + "\n".join(
                f"- #{s.id} «{s.name}» ({s.length()} работ) — "
                f"закрыта: {s.closed_reason or 'причина не указана'}"
                for s in state.series
                if not s.is_open()
            )
        return (
            "Текущей открытой серии нет. На этой итерации нужно открыть новую серию."
            + past
        )

    past = ""
    closed = [s for s in state.series if not s.is_open()]
    if closed:
        past = "\n\nПрошлые серии:\n" + "\n".join(
            f"- #{s.id} «{s.name}» ({s.length()} работ) — "
            f"закрыта: {s.closed_reason or 'причина не указана'}"
            for s in closed
        )

    # soft/hard hint
    hint = ""
    if cur.length() >= HARD_MAX_SERIES_LEN:
        hint = (
            f"\n\n⚠ ЖЁСТКИЙ ЛИМИТ: серия достигла {cur.length()} работ. "
            "На этой итерации series_status ДОЛЖЕН быть 'close' "
            "с осмысленным closed_reason и proposed_next_series. "
            "Если ты не закроешь серию сам, харнесс закроет её принудительно."
        )
    elif cur.length() >= SOFT_MAX_SERIES_LEN:
        hint = (
            f"\n\n⚠ МЯГКИЙ ЛИМИТ: серия достигла {cur.length()} работ "
            f"(рекомендуемый максимум — {SOFT_MAX_SERIES_LEN}). "
            "Скорее всего, пора закрывать её этой работой и предлагать следующую."
        )
    elif cur.length() < MIN_SERIES_LEN - 1:
        # "-1" because this iteration hasn't been added yet
        hint = (
            f"\n\nПримечание: серии ещё нет {MIN_SERIES_LEN} работ, "
            "закрыть её на этом шаге нельзя — харнесс отклонит."
        )

    return (
        f"Текущая серия: #{cur.id} «{cur.name}»\n"
        f"Тезис: {cur.thesis}\n"
        f"Длина серии к моменту этой итерации: {cur.length()} работ."
        f"{hint}{past}"
    )


def _full_memory(state: State, path_obj: dict | None) -> str:
    return (
        "# Состояние памяти\n\n"
        "## Серии\n\n"
        f"{_series_memory_block(state)}\n\n"
        "## artistic_path\n\n"
        f"{_path_memory_block(path_obj)}\n\n"
        "## Последние записи дневника\n\n"
        f"{_diary_memory_block(state)}\n"
    )


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #


@dataclass
class Ctx:
    n: int
    state: State
    path_obj: dict | None


def step1_create(ctx: Ctx) -> None:
    log.info("step 1/6 — create art.py (iteration %03d)", ctx.n)
    prev_art = read_text(
        ART_SCRIPT, default="(первая итерация, предыдущего скрипта нет)"
    )
    cur = ctx.state.current_series()
    cur_block = (
        f"Текущая серия: #{cur.id} «{cur.name}» — {cur.thesis}"
        if cur
        else "Серии пока нет — эта работа откроет новую."
    )
    user = (
        f"{_full_memory(ctx.state, ctx.path_obj)}\n\n"
        f"## Предыдущий art.py\n\n"
        f"```python\n{prev_art[:12000]}\n```\n\n"
        f"## Задача — Шаг 1 (итерация {ctx.n:03d})\n\n"
        f"{cur_block}\n\n"
        f"Напиши полный `art.py` для этой итерации. "
        f"Скрипт должен сохранить `art_{ctx.n:03d}.png` 1024×1024 в CWD. "
        f"В начале файла — обязательные комментарии "
        f"Iteration/Seed/Series/Preserved/Mutated.\n\n"
        f"Ответь **только исходным кодом Python** — без JSON, без пояснений, "
        f"без markdown-обёртки. Первая строка ответа должна быть первой "
        f"строкой файла `art.py`."
    )
    art_py = llm_text(
        user,
        temperature=0.95,
        max_tokens=12000,
        tag=f"step1_iter{ctx.n:03d}",
        expect_lang="python",
    )
    write_text(ART_SCRIPT, art_py)
    log.info("    wrote %s (%d bytes)", ART_SCRIPT.name, len(art_py))


def step2_render(ctx: Ctx) -> Path:
    log.info("step 2/6 — render")
    target = OUTPUT_DIR / f"art_{ctx.n:03d}.png"
    last_err = ""

    for attempt in range(1, MAX_FIX_RETRIES + 2):
        try:
            proc = subprocess.run(
                [ART_PYTHON, str(ART_SCRIPT)],
                cwd=str(OUTPUT_DIR),
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT,
            )
        except subprocess.TimeoutExpired as e:
            last_err = f"TimeoutExpired after {EXEC_TIMEOUT}s\n{e.stderr or ''}"
            log.warning("    art.py timed out (attempt %d)", attempt)
        else:
            if proc.returncode == 0 and target.exists():
                log.info("    rendered %s", target.name)
                return target
            last_err = (
                f"exit={proc.returncode}\n"
                f"stdout:\n{proc.stdout[-2000:]}\n"
                f"stderr:\n{proc.stderr[-4000:]}"
            )
            if proc.returncode == 0:
                last_err += f"\n(скрипт завершился успешно, но {target.name} не найден)"
            log.warning(
                "    art.py failed (attempt %d): %s", attempt, last_err.splitlines()[0]
            )

        if attempt > MAX_FIX_RETRIES:
            raise RuntimeError(
                f"art.py still failing after {MAX_FIX_RETRIES} fixes:\n{last_err}",
            )

        broken = read_text(ART_SCRIPT)
        fix_user = (
            f"Предыдущий `art.py` не отработал. Ошибка:\n\n"
            f"```\n{last_err}\n```\n\n"
            f"Текущий код:\n\n```python\n{broken}\n```\n\n"
            f"Почини его. Файл должен сохранить `art_{ctx.n:03d}.png` 1024×1024 в CWD.\n\n"
            f"Ответь **только исходным кодом Python** — без JSON, без пояснений, "
            f"без markdown-обёртки. Первая строка ответа должна быть первой "
            f"строкой файла `art.py`."
        )
        art_py = llm_text(
            fix_user,
            temperature=0.4,
            max_tokens=12000,
            tag=f"step2_fix_iter{ctx.n:03d}_att{attempt}",
            expect_lang="python",
        )
        write_text(ART_SCRIPT, art_py)
        log.info("    applied fix attempt %d", attempt)

    raise RuntimeError("unreachable")


def step3_observe(ctx: Ctx, png: Path) -> str:
    log.info("step 3/6 — observe %s", png.name)
    art_src = read_text(ART_SCRIPT)[:4000]
    user = [
        {
            "type": "text",
            "text": (
                f"{_full_memory(ctx.state, ctx.path_obj)}\n\n"
                f"## Задача — Шаг 3 (итерация {ctx.n:03d})\n\n"
                f"Посмотри на изображение. Верни JSON с одним полем "
                f"`observation` (5–8 строк, честное описание: композиция, "
                f"цвет, ритм, текстура, что получилось и что нет; не выдумывай "
                f"деталей).\n\n"
                f"Для справки — начало породившего кода:\n"
                f"```python\n{art_src}\n```"
            ),
        },
        {"type": "image_url", "image_url": {"url": encode_image(png)}},
    ]
    obj = llm_json(
        user,
        SCHEMA_STEP3,
        model=VISION_MODEL,
        temperature=0.6,
        max_tokens=2000,
        tag=f"step3_iter{ctx.n:03d}",
    )
    text = obj["observation"].strip()
    log.info("    observation: %s", text.splitlines()[0][:120] if text else "(empty)")
    return text


def step4_diary(ctx: Ctx, observation: str) -> dict:
    """Returns the validated diary entry (iteration-json shape)."""
    log.info("step 4/6 — diary entry")

    cur = ctx.state.current_series()
    cur_len = cur.length() if cur else 0

    user = (
        f"{_full_memory(ctx.state, ctx.path_obj)}\n\n"
        f"## Наблюдение (итерация {ctx.n:03d})\n\n{observation}\n\n"
        f"## Задача — Шаг 4\n\n"
        f"Верни JSON по схеме DiaryEntry. Поле `series_status`:\n"
        f"- 'continue' — работа продолжает текущую серию;\n"
        f"- 'close' — серия завершается этой работой. Разрешено только если "
        f"в серии уже ≥{MIN_SERIES_LEN} работ.\n"
        f"При `close` обязательны содержательные `closed_reason` "
        f"и `proposed_next_series`. Если текущей серии ещё нет "
        f"(начало или принудительное закрытие предыдущей), `series_status` "
        f"всё равно ставь 'continue' — харнесс сам откроет новую серию "
        f"по твоему `proposed_next_series`, так что лучше заполни его и тут."
    )

    obj = llm_json(
        user,
        SCHEMA_STEP4,
        temperature=0.85,
        max_tokens=2000,
        tag=f"step4_iter{ctx.n:03d}",
    )

    # sanity: non-empty fields
    required = ("title", "what_i_see", "what_resonates", "what_is_silent", "where_next")
    for k in required:
        if not obj.get(k) or len(obj[k].strip()) < 10:
            raise RuntimeError(
                f"diary field {k!r} is empty or too short: {obj.get(k)!r}"
            )

    # enforce series policy
    status = obj["series_status"]
    if status == "close" and cur is not None and cur_len < MIN_SERIES_LEN:
        log.warning(
            "    model wanted to close series #%d after only %d works; "
            "overriding to 'continue' (MIN_SERIES_LEN=%d)",
            cur.id,
            cur_len,
            MIN_SERIES_LEN,
        )
        obj["series_status"] = "continue"
        obj["closed_reason"] = None

    if status == "close":
        if not obj.get("closed_reason"):
            obj["closed_reason"] = (
                obj.get("where_next") or "Серия исчерпала свой голос."
            )
        if not obj.get("proposed_next_series"):
            log.warning(
                "    close without proposed_next_series; "
                "will ask for a new series name next step."
            )

    return obj


def step4b_series_retrospective(
    state: State, series: Series, path_obj: dict | None
) -> dict | None:
    """Show the model every image of the just-closed series and ask for an
    arc-level retrospective. Returns the retrospective dict, or None if the
    call failed (loop continues either way)."""
    log.info(
        "step 4b/6 — retrospective for series #%d «%s» (%d works)",
        series.id,
        series.name,
        series.length(),
    )

    # gather images + per-iteration summaries in order
    entries: list[tuple[int, dict, Path]] = []
    for n in series.iterations:
        it = state.iterations.get(n)
        if it is None:
            continue
        png = OUTPUT_DIR / it.get("image_path", f"art_{n:03d}.png")
        if not png.exists():
            log.warning("    missing image %s, skipping from retrospective", png.name)
            continue
        entries.append((n, it, png))

    if len(entries) < 2:
        log.warning(
            "    series #%d has only %d renderable works; skipping retrospective",
            series.id,
            len(entries),
        )
        return None

    # build the message: one text header + inline image/caption pairs
    header = (
        f"{_full_memory(state, path_obj)}\n\n"
        f"## Задача — Шаг 4b (Ретроспектива серии)\n\n"
        f"Серия #{series.id} «{series.name}» только что закрылась "
        f"на итерации {series.closed_at}. Её тезис был:\n"
        f"> {series.thesis}\n\n"
        f"Причина закрытия: {series.closed_reason or 'не указана'}.\n\n"
        f"Ниже — все {len(entries)} работ серии по порядку, "
        f"в уменьшенном виде (512px). Перед каждым изображением — "
        f"твоя же дневниковая запись о нём.\n\n"
        f"Посмотри на серию как на единое произведение. "
        f"Верни JSON SeriesRetrospective — поэтическое имя всей серии "
        f"(может совпадать с «{series.name}» или уточнять его), "
        f"3–5 предложений об арке, что проявилось, что разочаровало, "
        f"короткий вердикт, и мост к следующей серии."
    )

    content: list[dict] = [{"type": "text", "text": header}]
    for n, it, png in entries:
        caption = (
            f"Итерация {n:03d} — «{it['title']}»\n"
            f"Что я видел тогда: {it['what_i_see']}\n"
            f"Что резонировало: {it['what_resonates']}\n"
            f"Что молчало: {it['what_is_silent']}"
        )
        content.append({"type": "text", "text": caption})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_downscaled(png, size=512)},
            }
        )

    try:
        obj = llm_json(
            content,
            SCHEMA_RETRO,
            model=VISION_MODEL,
            temperature=0.7,
            max_tokens=3000,
            tag=f"step4b_series{series.id:02d}",
        )
    except Exception:
        log.exception("    retrospective call failed; leaving series without it")
        return None

    # validate non-empty
    required = (
        "title",
        "arc_summary",
        "what_emerged",
        "what_disappointed",
        "verdict",
        "bridge_to_next",
    )
    for k in required:
        if not obj.get(k) or len(obj[k].strip()) < 10:
            log.warning(
                "    retrospective field %r is empty/too short: %r", k, obj.get(k)
            )
            return None

    log.info("    retrospective title: «%s»", obj["title"])
    return obj


def step5_path(ctx: Ctx, observation: str, force: bool = False) -> dict | None:
    """Returns path_obj if rewritten, else None."""
    first_time = ctx.path_obj is None
    due = force or ctx.n == 1 or ctx.n % PATH_REWRITE_EVERY == 0
    if not (first_time or due):
        log.info("step 5/6 — path (skipped)")
        return None

    log.info("step 5/6 — rewrite artistic_path")
    is_first = ctx.n == 1 and first_time
    user = f"{_full_memory(ctx.state, ctx.path_obj)}\n\n## Задача — Шаг 5\n\n" + (
        f"Это итерация 001. Верни JSON ArtisticPath: поле `opening` "
        f"заполни одной-двумя фразами, остальные поля заполни "
        f"лаконичными набросками."
        if is_first
        else "Перепиши artistic_path целиком. Верни JSON ArtisticPath. "
        "Поле `opening` ставь null. Остальные семь полей — плотный срез "
        "(150–300 слов суммарно)."
    )

    budget = 1000 if is_first else 3500
    obj = llm_json(
        user,
        SCHEMA_STEP5,
        temperature=0.7,
        max_tokens=budget,
        tag=f"step5_iter{ctx.n:03d}",
    )
    save_path_json(obj)
    return obj


# --------------------------------------------------------------------------- #
# renderers (pure: JSON state -> markdown views)
# --------------------------------------------------------------------------- #


def render_diary(state: State) -> str:
    if not state.series:
        return ""
    parts: list[str] = ["# Выставочный дневник\n"]
    for s in state.series:
        header = f"## Серия {s.id} — «{s.name}»"
        if not s.is_open():
            header += "  *(закрыта)*"
        parts.append(header)
        parts.append(f"*{s.thesis}*\n")
        for n in s.iterations:
            it = state.iterations.get(n)
            if it is None:
                continue
            parts.append(f"### Итерация {n:03d} — «{it['title']}»")
            parts.append(f"- **Что я вижу.** {it['what_i_see']}")
            parts.append(f"- **Что звучит.** {it['what_resonates']}")
            parts.append(f"- **Что молчит.** {it['what_is_silent']}")
            parts.append(f"- **Куда дальше.** {it['where_next']}")
            parts.append("")
            parts.append("---")
            parts.append("")
        # Series footer: retrospective is preferred; fall back to a short line.
        if not s.is_open():
            if s.retrospective:
                r = s.retrospective
                parts.append(f"### Взгляд назад — «{r['title']}»")
                parts.append("")
                parts.append(f"> {r['arc_summary']}")
                parts.append("")
                parts.append(f"- **Что проявилось.** {r['what_emerged']}")
                parts.append(f"- **Что разочаровало.** {r['what_disappointed']}")
                parts.append(f"- **Вердикт.** {r['verdict']}")
                parts.append(f"- **Мост к следующему.** {r['bridge_to_next']}")
                parts.append("")
                parts.append(f"*Серия закрыта на итерации {s.closed_at}.*\n")
            elif s.closed_reason:
                parts.append(
                    f"*Серия закрыта на итерации {s.closed_at}: {s.closed_reason}*\n",
                )
    return "\n".join(parts).rstrip() + "\n"


def render_path(path_obj: dict | None) -> str:
    if not path_obj:
        return ""
    sections = [
        ("opening", "Начальная мысль"),
        ("what_attracts_me", "Что меня притягивает"),
        ("what_i_dislike", "Что мне не нравится"),
        ("what_works", "Что у меня получается"),
        ("where_i_stumble", "Где я спотыкаюсь"),
        ("emerging_language", "Мой формирующийся язык"),
        ("current_series", "Текущая серия"),
        ("next_hypothesis", "Следующая гипотеза"),
    ]
    lines = ["# Путь художника\n"]
    for key, label in sections:
        val = path_obj.get(key)
        if not val:
            continue
        lines.append(f"## {label}")
        lines.append(val.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def refresh_markdown(state: State, path_obj: dict | None) -> None:
    write_text(DIARY_PATH, render_diary(state))
    write_text(PATH_MD, render_path(path_obj))


# --------------------------------------------------------------------------- #
# main cycle
# --------------------------------------------------------------------------- #


def _ensure_open_series(
    state: State, at_iter: int, proposal: dict | None = None
) -> Series:
    """Open a new series at `at_iter` if none is currently open."""
    cur = state.current_series()
    if cur is not None:
        return cur
    if proposal and proposal.get("name") and proposal.get("thesis"):
        return state.open_series(proposal["name"], proposal["thesis"], at_iter)
    # no proposal — ask the model in a tiny targeted call
    log.warning("    no open series and no proposal; asking model to name one")
    user = (
        f"{_full_memory(state, load_path_json())}\n\n"
        "## Задача\n\nТекущей открытой серии нет, а харнессу она нужна. "
        "Предложи имя и тезис новой серии, которая осмысленно следует из "
        "прошлого пути. Верни JSON с полями `name` и `thesis`."
    )
    schema = {
        "name": "NewSeries",
        "strict": True,
        "schema": _obj({"name": {"type": "string"}, "thesis": {"type": "string"}}),
    }
    obj = llm_json(
        user,
        schema,
        temperature=0.7,
        max_tokens=600,
        tag=f"open_series_iter{at_iter:03d}",
    )
    return state.open_series(obj["name"], obj["thesis"], at_iter)


def one_cycle(state: State) -> None:
    n = state.next_iteration_number()
    path_obj = load_path_json()
    ctx = Ctx(n=n, state=state, path_obj=path_obj)
    log.info("=== iteration %03d ===", n)

    # ensure we have an open series BEFORE step 1 so its metadata can bake
    # into the art.py header. On iter 1 we don't have a proposal yet — open
    # a provisional series whose name/thesis comes from the path (if any) or
    # the model itself.
    if state.current_series() is None:
        _ensure_open_series(state, at_iter=n)

    step1_create(ctx)
    png = step2_render(ctx)
    observation = step3_observe(ctx, png)
    diary = step4_diary(ctx, observation)

    # record this iteration against the current series
    cur = state.current_series()
    assert cur is not None
    iter_json = {
        "iteration": n,
        "series_id": cur.id,
        "series_name": cur.name,
        "title": diary["title"],
        "what_i_see": diary["what_i_see"],
        "what_resonates": diary["what_resonates"],
        "what_is_silent": diary["what_is_silent"],
        "where_next": diary["where_next"],
        "series_status": diary["series_status"],
        "closed_reason": diary.get("closed_reason"),
        "proposed_next_series": diary.get("proposed_next_series"),
        "image_path": png.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    state.save_iteration(iter_json)
    state.append_to_current_series(n)

    # apply series policy: close if model said so, OR if we hit HARD cap
    closed_this_cycle = False
    closed_series_ref: Series | None = None
    if diary["series_status"] == "close":
        reason = diary.get("closed_reason") or "Серия завершена художником."
        state.close_series(at_iter=n, reason=reason)
        closed_series_ref = cur
        closed_this_cycle = True
    elif cur.length() >= HARD_MAX_SERIES_LEN:
        reason = (
            "Жёсткий лимит серии достигнут; харнесс закрыл её "
            "принудительно, следующая работа откроет новую."
        )
        state.close_series(at_iter=n, reason=reason)
        closed_series_ref = cur
        closed_this_cycle = True

    # Шаг 4b — retrospective on the just-closed series (before opening next)
    if closed_this_cycle and closed_series_ref is not None:
        retro = step4b_series_retrospective(state, closed_series_ref, path_obj)
        if retro is not None:
            state.set_series_retrospective(closed_series_ref.id, retro)

    # if closed, try to immediately open the next series using the proposal
    if closed_this_cycle:
        proposal = diary.get("proposed_next_series")
        # iter N is the last of the closed series; the NEW series opens at N+1
        _ensure_open_series(state, at_iter=n + 1, proposal=proposal)

    # step 5 — rewrite path. Force-rewrite on series boundary.
    new_path = step5_path(ctx, observation, force=closed_this_cycle)
    if new_path is not None:
        path_obj = new_path

    # always refresh markdown views
    refresh_markdown(state, path_obj)
    log.info("=== iteration %03d done ===", n)


def main() -> None:
    log.info(
        "driver starting. model=%s vision=%s workdir=%s",
        MODEL,
        VISION_MODEL,
        OUTPUT_DIR,
    )
    state = State()
    log.info(
        "loaded state: %d series, %d iterations",
        len(state.series),
        len(state.iterations),
    )
    while True:
        try:
            one_cycle(state)
        except KeyboardInterrupt:
            log.info("interrupted by user, bye")
            return
        except Exception:
            log.exception("cycle failed, sleeping 5s and continuing")
            time.sleep(5)


if __name__ == "__main__":
    main()
