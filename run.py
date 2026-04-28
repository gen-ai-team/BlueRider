# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.40",
#     "python-dotenv>=1.0",
#     "pillow>=10",
#     "numpy>=1.24",
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
import numpy as np

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
PROMPT_PATH = REPO_ROOT / "prompt.md"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", REPO_ROOT / "output")).resolve()

ART_SCRIPT = OUTPUT_DIR / "art.py"
SCRIPTS_DIR = OUTPUT_DIR / "scripts"
DIARY_PATH = OUTPUT_DIR / "exhibition_diary.md"
PATH_MD = OUTPUT_DIR / "artistic_path.md"
LOG_PATH = OUTPUT_DIR / "run.log"
DEBUG_DIR = OUTPUT_DIR / "debug"

STATE_DIR = OUTPUT_DIR / "state"
SERIES_JSON = STATE_DIR / "series.json"
PATH_JSON = STATE_DIR / "path.json"
LIBRARY_JSON = STATE_DIR / "library.json"
ITER_DIR = STATE_DIR / "iterations"
LIBRARY_MD = OUTPUT_DIR / "library.md"

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
VISION_MODEL = os.getenv("VISION_MODEL", MODEL)
EXEC_TIMEOUT = int(os.getenv("EXEC_TIMEOUT", "240"))
MAX_FIX_RETRIES = int(os.getenv("MAX_FIX_RETRIES", "3"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))

# series policy
MIN_SERIES_LEN = int(os.getenv("MIN_SERIES_LEN", "7"))
SOFT_MAX_SERIES_LEN = int(os.getenv("SOFT_MAX_SERIES_LEN", "12"))
HARD_MAX_SERIES_LEN = int(os.getenv("HARD_MAX_SERIES_LEN", "15"))
# Expected total number of series for the whole run. Drives the
# explore/exploit framing the model sees — early on it should build its
# library, later on it should lean on it to create a recognizable body of
# work. 0 disables progress framing (infinite / unknown run).
EXPECTED_SERIES = int(os.getenv("EXPECTED_SERIES", "12"))

# Rendered image resolution. Injected into the prompt so the LLM doesn't
# hardcode a mismatched number. Change only if you also change consumers
# (e.g. make_video.py).
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "1024"))

# Perceptual-similarity bucket thresholds. Used both to bucket harness-side
# measurements and surfaced into the prompt so the model and the harness
# stay in sync.
SIM_STRONG_REPEAT = float(os.getenv("SIM_STRONG_REPEAT", "0.80"))
SIM_MILD_REPEAT = float(os.getenv("SIM_MILD_REPEAT", "0.60"))
# path.json used to be rewritten every N iterations. That turned into an echo
# chamber (the model only ever ratcheted the current tendency tighter). Now
# path.json is rewritten only on series boundaries (close/open) and at very
# first iteration. Setting PATH_REWRITE_EVERY>0 re-enables the periodic rewrite
# for backward compatibility, but the default is "only on boundary".
PATH_REWRITE_EVERY = int(os.getenv("PATH_REWRITE_EVERY", "0"))

# how many recent images we show to step 1 as visual memory of the current
# series (in addition to the single latest one). 0 disables the survey.
STEP1_SERIES_SURVEY = int(os.getenv("STEP1_SERIES_SURVEY", "2"))
# how many past-series "last frame" images we show to step 1 when the iter is
# the first of a new series (to force cross-series visual differentiation).
STEP1_PAST_SERIES_SURVEY = int(os.getenv("STEP1_PAST_SERIES_SURVEY", "4"))
# rolling window over which the harness computes visual-similarity scores and
# surfaces them to steps 3/4/1 so the model can react to repetition.
SIMILARITY_WINDOW = int(os.getenv("SIMILARITY_WINDOW", "6"))


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
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

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

SYSTEM_PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")


def _build_system_prompt() -> str:
    """Substitute the handful of config values that the prompt references
    verbatim. Keeps numbers in run.py and prompt.md from drifting."""
    subs = {
        "IMAGE_SIZE": str(IMAGE_SIZE),
        "MIN_SERIES_LEN": str(MIN_SERIES_LEN),
        "SOFT_MAX_SERIES_LEN": str(SOFT_MAX_SERIES_LEN),
        "HARD_MAX_SERIES_LEN": str(HARD_MAX_SERIES_LEN),
        "SIM_STRONG_REPEAT": f"{SIM_STRONG_REPEAT:.2f}",
        "SIM_MILD_REPEAT": f"{SIM_MILD_REPEAT:.2f}",
        "EXPECTED_SERIES": str(EXPECTED_SERIES) if EXPECTED_SERIES > 0 else "—",
    }
    out = SYSTEM_PROMPT_TEMPLATE
    for k, v in subs.items():
        out = out.replace("{{" + k + "}}", v)
    # Sanity: if any unresolved {{FOO}} template tokens remain, surface it —
    # almost certainly a typo rather than legitimate prose.
    leftovers = re.findall(r"\{\{[A-Z_]+\}\}", out)
    if leftovers:
        log.warning("prompt has unresolved template tokens: %s", leftovers)
    return out


SYSTEM_PROMPT = _build_system_prompt()


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
    "schema": _obj(
        {
            # --- enum-constrained axes (machine-comparable between iterations) ---
            "composition": {
                "type": "string",
                "enum": [
                    "centered",
                    "rule_of_thirds",
                    "grid",
                    "scattered",
                    "field",
                    "diagonal",
                    "spiral",
                    "radial",
                    "layered",
                    "other",
                ],
            },
            "shape_family": {
                "type": "string",
                "enum": [
                    "circle",
                    "line",
                    "polygon",
                    "curve",
                    "blob",
                    "noise",
                    "lattice",
                    "dot",
                    "mixed",
                    "other",
                ],
            },
            "density": {
                "type": "string",
                "enum": ["empty", "sparse", "medium", "dense", "saturated"],
            },
            "symmetry": {
                "type": "string",
                "enum": [
                    "none",
                    "bilateral",
                    "radial",
                    "rotational",
                    "translational",
                    "mixed",
                ],
            },
            "element_count": {
                "type": "string",
                "enum": ["1", "2-5", "6-20", "20-100", ">100", "field"],
            },
            "negative_space": {
                "type": "string",
                "enum": ["none", "low", "medium", "high", "dominant"],
            },
            "uses_blur_or_glow": {"type": "boolean"},
            "similarity_to_previous": {
                "type": "string",
                "enum": ["novel", "mild_repeat", "strong_repeat"],
                "description": "How close this work is to the immediately "
                "preceding one (palette + composition + shape_family + "
                "density together). 'novel' = clearly different along at "
                "least one axis; 'strong_repeat' = could be mistaken for it.",
            },
            # --- free-text fields (what enums can't capture) ---
            "palette": {
                "type": "string",
                "description": "2–4 colours that carry the piece, named or "
                "with approximate hex, plus their role "
                "(background/dominant/accent/etc). Max ~30 words.",
            },
            "observation": {
                "type": "string",
                "description": "5–8 строк честного описания того, что ты "
                "видишь — композиция, ритм, текстура, что получилось, что "
                "нет. Опирайся и на картинку, и на знание кода.",
            },
            "what_works": {
                "type": "string",
                "description": "1–2 sentences: the single strongest thing "
                "about this image.",
            },
            "what_fails": {
                "type": "string",
                "description": "1–2 sentences: the single weakest thing, or "
                "what feels accidental / unintended.",
            },
        },
        required=[
            "composition",
            "shape_family",
            "density",
            "symmetry",
            "element_count",
            "negative_space",
            "uses_blur_or_glow",
            "similarity_to_previous",
            "palette",
            "observation",
            "what_works",
            "what_fails",
        ],
    ),
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
            # explicit acknowledgement of the harness-computed similarity
            # score for this image vs recent ones. Forces the model to react
            # to visual repetition rather than verbally paper over it.
            "repetition_check": {
                "type": "string",
                "enum": ["novel", "mild_repeat", "strong_repeat"],
            },
            "repetition_note": {
                "type": "string",
                "description": "1 sentence. If strong_repeat/mild_repeat — "
                "what exactly is repeating (palette? composition? same "
                "silhouette?) and what the next iteration will change to "
                "break out. If novel — what was the concrete change that "
                "made it novel.",
            },
            "series_status": {"type": "string", "enum": ["continue", "close"]},
            "closed_reason": {"type": ["string", "null"]},
            "proposed_next_series": {
                "anyOf": [
                    {"type": "null"},
                    _obj(
                        {
                            "name": {"type": "string"},
                            "thesis": {"type": "string"},
                            # force explicit commitment to what makes the new
                            # series visually different from everything done
                            # before. "verbal difference" alone is not enough.
                            "must_differ_from_previous_in": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "palette",
                                        "composition",
                                        "shape_family",
                                        "density",
                                        "symmetry",
                                        "algorithmic_kernel",
                                        "scale",
                                        "texture",
                                    ],
                                },
                                "minItems": 2,
                                "description": "At least two axes on which "
                                "the new series must visibly break with "
                                "prior series. Not a wish — a contract.",
                            },
                            "forbidden_features": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Concrete things the new "
                                "series must NOT contain (e.g. 'ochre-on-"
                                "indigo palette', 'central medallion', "
                                "'gaussian glow', 'voronoi rendering'). "
                                "Each item one short phrase.",
                            },
                        },
                        required=[
                            "name",
                            "thesis",
                            "must_differ_from_previous_in",
                            "forbidden_features",
                        ],
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
            "repetition_check",
            "repetition_note",
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
            # Library curation happens in two steps: first the model commits
            # to a verb ("nothing" / "add" / "retire" / "add_and_retire"),
            # then it fills only the slots that verb allows. The enum is
            # there so that "do not touch the library" becomes a first-class
            # choice the model has to type out — not a quiet empty array.
            "library_decision": {
                "type": "string",
                "enum": ["nothing", "add", "retire", "add_and_retire"],
                "description": (
                    "Что ты собираешься сделать с библиотекой. По умолчанию "
                    "— 'nothing': большинство серий ничего не добавляют и "
                    "ничего не списывают, и это нормально. 'add' — только "
                    "если в этой серии появился действительно отдельный "
                    "приём, к которому ты захочешь вернуться; 'retire' — "
                    "если какая-то старая запись перестала звучать твоей; "
                    "'add_and_retire' — редкий случай замены одной записи "
                    "на более точную."
                ),
            },
            "nothing_reason": {
                "type": ["string", "null"],
                "description": (
                    "Если library_decision='nothing' — одно предложение, "
                    "почему ты решил ничего не трогать (например: «в этой "
                    "серии я просто развивал уже записанный приём», «я ещё "
                    "не уверен, удержится ли этот жест»). Иначе null."
                ),
            },
            "library_additions": {
                "type": "array",
                "items": _obj(
                    {
                        "name": {
                            "type": "string",
                            "description": (
                                "Короткое имя техники/приёма (3–7 слов). "
                                "Не начинай подряд все записи с 'Метод …' / "
                                "'Алгоритм …' — если в библиотеке уже есть "
                                "такие, выбери другую форму имени."
                            ),
                        },
                        "when_to_use": {
                            "type": "string",
                            "description": "Одно-два предложения: в каком "
                            "художественном контексте этот приём стоит "
                            "доставать.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Свободное описание техники — "
                            "что она делает, чем отличается, как её "
                            "настраивать. Русский язык.",
                        },
                        "code_snippet": {
                            "type": ["string", "null"],
                            "description": (
                                "Либо null (если приём словесный и код "
                                "ничего не добавит), либо полноценный "
                                "рабочий фрагмент Python. Тело функции "
                                "должно делать то, что описывает имя — "
                                "никаких заглушек вроде `pass`, "
                                "`# simplified logic`, `# conceptual "
                                "snippet`, `...`: если ты не готов "
                                "написать реализацию, ставь null. Минимум "
                                "~200 символов осмысленного кода. Код "
                                "пишется под окружение art.py (numpy, "
                                "PIL/Pillow, scipy, noise + stdlib)."
                            ),
                        },
                    },
                    required=[
                        "name",
                        "when_to_use",
                        "description",
                        "code_snippet",
                    ],
                ),
                "description": (
                    "Заполняй только если library_decision='add' или "
                    "'add_and_retire'. В остальных случаях — пустой массив. "
                    "Обычно ноль записей; если add — одна запись, очень "
                    "редко две. Если уже есть похожая запись — лучше "
                    "списать её через retire с superseded_by, чем плодить "
                    "дубликат."
                ),
            },
            "library_retirements": {
                "type": "array",
                "items": _obj(
                    {
                        "entry_id": {
                            "type": "integer",
                            "description": "id записи из библиотеки, "
                            "которую ты списываешь.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Одно предложение: почему этот "
                            "приём больше не твой.",
                        },
                        "superseded_by": {
                            "type": ["integer", "null"],
                            "description": "Если списание — это замена на "
                            "более точную запись из library_additions той "
                            "же ретроспективы — поставь 0 (харнесс "
                            "подставит реальный id) или null.",
                        },
                    },
                    required=["entry_id", "reason", "superseded_by"],
                ),
                "description": (
                    "Заполняй только если library_decision='retire' или "
                    "'add_and_retire'. В остальных случаях — пустой массив."
                ),
            },
        },
        required=[
            "title",
            "arc_summary",
            "what_emerged",
            "what_disappointed",
            "verdict",
            "bridge_to_next",
            "library_decision",
            "nothing_reason",
            "library_additions",
            "library_retirements",
        ],
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
    # Cross-series contract set at series open: which axes this series
    # commits to differ from every prior series on, and which concrete
    # features are explicitly forbidden. Forces real differentiation rather
    # than verbal repainting.
    must_differ_from_previous_in: list[str] = field(default_factory=list)
    forbidden_features: list[str] = field(default_factory=list)

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
            "must_differ_from_previous_in": list(self.must_differ_from_previous_in),
            "forbidden_features": list(self.forbidden_features),
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
            must_differ_from_previous_in=list(
                d.get("must_differ_from_previous_in", [])
            ),
            forbidden_features=list(d.get("forbidden_features", [])),
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
    def open_series(
        self,
        name: str,
        thesis: str,
        at_iter: int,
        *,
        must_differ_from_previous_in: list[str] | None = None,
        forbidden_features: list[str] | None = None,
    ) -> Series:
        new_id = (self.series[-1].id + 1) if self.series else 1
        s = Series(
            id=new_id,
            name=name,
            thesis=thesis,
            opened_at=at_iter,
            must_differ_from_previous_in=list(must_differ_from_previous_in or []),
            forbidden_features=list(forbidden_features or []),
        )
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
# technique library
# --------------------------------------------------------------------------- #
# A tiny key-value style store the agent curates across the run. Entries are
# added only at series-close (step 4b) and only when the model feels strongly
# about saving something. Each entry is free-form: prose describing a
# technique/palette/approach, optionally with a python snippet that should
# be drop-in usable under the art.py environment (numpy, pillow, scipy,
# noise + stdlib). The model can also retire / supersede old entries when
# they no longer represent its voice, so the library stays curated rather
# than growing into a landfill.


@dataclass
class LibraryEntry:
    id: int
    name: str
    when_to_use: str
    description: str
    code_snippet: str | None
    created_at: str
    created_in_series: int | None
    retired: bool = False
    retired_at: str | None = None
    retired_reason: str | None = None
    superseded_by: int | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "when_to_use": self.when_to_use,
            "description": self.description,
            "code_snippet": self.code_snippet,
            "created_at": self.created_at,
            "created_in_series": self.created_in_series,
            "retired": self.retired,
            "retired_at": self.retired_at,
            "retired_reason": self.retired_reason,
            "superseded_by": self.superseded_by,
        }

    @staticmethod
    def from_dict(d: dict) -> "LibraryEntry":
        return LibraryEntry(
            id=d["id"],
            name=d["name"],
            when_to_use=d.get("when_to_use", ""),
            description=d.get("description", ""),
            code_snippet=d.get("code_snippet"),
            created_at=d.get("created_at", ""),
            created_in_series=d.get("created_in_series"),
            retired=bool(d.get("retired", False)),
            retired_at=d.get("retired_at"),
            retired_reason=d.get("retired_reason"),
            superseded_by=d.get("superseded_by"),
        )


class Library:
    """Persistent view over state/library.json."""

    def __init__(self) -> None:
        self.entries: list[LibraryEntry] = self._load()

    @staticmethod
    def _load() -> list[LibraryEntry]:
        if not LIBRARY_JSON.exists():
            return []
        data = json.loads(LIBRARY_JSON.read_text(encoding="utf-8"))
        return [LibraryEntry.from_dict(e) for e in data.get("entries", [])]

    def save(self) -> None:
        LIBRARY_JSON.write_text(
            json.dumps(
                {"entries": [e.to_dict() for e in self.entries]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def active(self) -> list[LibraryEntry]:
        return [e for e in self.entries if not e.retired]

    def by_id(self, entry_id: int) -> LibraryEntry | None:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None

    def add(
        self,
        *,
        name: str,
        when_to_use: str,
        description: str,
        code_snippet: str | None,
        series_id: int | None,
    ) -> LibraryEntry:
        new_id = (max((e.id for e in self.entries), default=0)) + 1
        e = LibraryEntry(
            id=new_id,
            name=name.strip(),
            when_to_use=when_to_use.strip(),
            description=description.strip(),
            code_snippet=(code_snippet.strip() if code_snippet else None) or None,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            created_in_series=series_id,
        )
        self.entries.append(e)
        self.save()
        return e

    def retire(
        self, entry_id: int, *, reason: str, superseded_by: int | None = None
    ) -> LibraryEntry | None:
        e = self.by_id(entry_id)
        if e is None or e.retired:
            return None
        e.retired = True
        e.retired_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        e.retired_reason = (reason or "").strip() or None
        e.superseded_by = superseded_by
        self.save()
        return e


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


# --------------------------------------------------------------------------- #
# perceptual similarity (pillow+numpy only)
# --------------------------------------------------------------------------- #
# Fingerprint = (dhash_bits, color_histogram). We store them on the iter_json
# so we never have to recompute for historical images. Comparing is cheap:
#   dhash   : 16×16 grayscale → horizontal-gradient bits (256 bits), hamming
#             distance normalized to [0,1]; 0 means identical structure.
#   colhist : 4×4×4 RGB histogram (64 bins) normalized; L1 distance / 2 in
#             [0,1]; 0 means identical colour distribution.
# "similarity" in [0,1] is 1 - weighted distance.


def _dhash_bits(img: Image.Image, size: int = 16) -> str:
    """Return a 256-bit hex string (difference hash). `size` controls bits."""
    gs = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(gs, dtype=np.int16)
    diff = a[:, 1:] > a[:, :-1]  # shape (size, size)
    bits = np.packbits(diff.flatten())
    return bits.tobytes().hex()


def _color_hist(img: Image.Image, bins_per_channel: int = 4) -> list[float]:
    """Return a normalized 4×4×4 = 64-bin RGB histogram as a list of floats."""
    small = img.convert("RGB").resize((96, 96), Image.LANCZOS)
    a = np.asarray(small, dtype=np.uint8)
    # quantize each channel to `bins_per_channel`
    b = (a.astype(np.int32) * bins_per_channel // 256).clip(0, bins_per_channel - 1)
    flat = b[..., 0] * bins_per_channel * bins_per_channel + (
        b[..., 1] * bins_per_channel + b[..., 2]
    )
    hist = np.bincount(flat.ravel(), minlength=bins_per_channel**3).astype(np.float32)
    total = float(hist.sum()) or 1.0
    return (hist / total).tolist()


def compute_fingerprint(path: Path) -> dict:
    with Image.open(path) as im:
        return {
            "dhash": _dhash_bits(im),
            "colhist": _color_hist(im),
        }


def _hamming_dist_hex(a: str, b: str) -> int:
    """Hamming distance in bits between two equal-length hex strings."""
    if len(a) != len(b):
        return max(len(a), len(b)) * 4
    xa = int(a, 16)
    xb = int(b, 16)
    return bin(xa ^ xb).count("1")


def _l1_dist(a: list[float], b: list[float]) -> float:
    la, lb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if la.shape != lb.shape:
        return 1.0
    return float(np.abs(la - lb).sum() / 2.0)  # in [0,1]


def similarity(fp_a: dict | None, fp_b: dict | None) -> float:
    """Return a similarity score in [0,1]. 1.0 = identical.

    Weights structural (dhash) and colour (histogram) distances equally.
    None fingerprints → 0.0 (treat as totally different / unknown).
    """
    if not fp_a or not fp_b:
        return 0.0
    dh_bits = 16 * 16  # 256 bits from our dhash
    dh_raw = _hamming_dist_hex(fp_a["dhash"], fp_b["dhash"])
    dh = dh_raw / dh_bits  # 0..1
    ch = _l1_dist(fp_a["colhist"], fp_b["colhist"])  # 0..1
    dist = 0.5 * dh + 0.5 * ch
    return max(0.0, min(1.0, 1.0 - dist))


def similarity_bucket(sim: float) -> str:
    """Map a similarity score to the enum used in step-3 / step-4 schemas."""
    if sim >= SIM_STRONG_REPEAT:
        return "strong_repeat"
    if sim >= SIM_MILD_REPEAT:
        return "mild_repeat"
    return "novel"


# --------------------------------------------------------------------------- #
# art.py provenance helpers
# --------------------------------------------------------------------------- #


_PRESERVED_RE = re.compile(r"^#\s*Preserved:\s*(.+)$", re.MULTILINE)
_MUTATED_RE = re.compile(r"^#\s*Mutated:\s*(.+)$", re.MULTILINE)


def parse_art_py_metadata(src: str) -> dict:
    """Pull `Preserved:` and `Mutated:` comment lines out of an art.py header.
    Returns empty strings when absent."""
    pres = _PRESERVED_RE.search(src)
    muts = _MUTATED_RE.search(src)
    return {
        "preserved": pres.group(1).strip() if pres else "",
        "mutated": muts.group(1).strip() if muts else "",
    }


def archive_art_script(n: int, src: str) -> Path:
    """Persist this iteration's script as scripts/art_NNN.py alongside the
    rendered PNG. The top-level art.py stays as the "latest" for readability
    and as the source the fixer reads on retry."""
    target = SCRIPTS_DIR / f"art_{n:03d}.py"
    target.write_text(src, encoding="utf-8")
    return target


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
        "temperature": temperature,
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


def _progress_block(state: State) -> str:
    """Tell the model roughly where it is on the planned arc, and nudge it
    from 'explore' toward 'exploit' as the run progresses. When
    EXPECTED_SERIES is 0 we skip this block (unknown horizon).
    """
    if EXPECTED_SERIES <= 0:
        return ""
    closed = [s for s in state.series if not s.is_open()]
    completed = len(closed)
    cur = state.current_series()
    in_progress = 1 if cur is not None else 0
    # "index of the series currently being worked on" is closed+1 while cur
    # is open; we show it as a 1-based position.
    position = completed + in_progress
    position = max(1, position)
    frac = min(1.0, position / EXPECTED_SERIES)
    # three soft phases, just as a framing for the model
    if frac < 0.33:
        phase = (
            "РАННЯЯ ФАЗА — исследуй широко. Почти в каждой серии меняй "
            "алгоритмическое ядро / палитровую логику / композиционную "
            "грамматику. Библиотеку приёмов собирай скупо — записывай только "
            "то, что реально хочется вытащить потом."
        )
    elif frac < 0.70:
        phase = (
            "СРЕДНЯЯ ФАЗА — начинай консолидироваться. Уже видно, что именно "
            "твоё; пробуй соединять находки между сериями. Новые записи в "
            "библиотеку — только если правда открылся отдельный приём, а не "
            "вариация уже записанного; при случае списывай записи, которые "
            "перестали звучать твоими."
        )
    else:
        phase = (
            "ПОЗДНЯЯ ФАЗА — пора эксплуатировать. Работа должна быть "
            "узнаваемо твоей: опирайся на собранную библиотеку приёмов и "
            "сложившийся язык. При этом не коллапсируй в одну картинку — "
            "каждая серия всё равно обязана отличаться от предыдущих по ≥2 "
            "осям. Новые записи в библиотеку — почти не нужны; списание "
            "устаревшего — нормально."
        )
    return (
        f"Серия #{position} из ~{EXPECTED_SERIES} запланированных "
        f"(завершено: {completed}).\n"
        f"{phase}"
    )


def _library_memory_block(library: Library) -> str:
    """Compact view of the active technique library. Hidden entirely when
    empty, so the model doesn't feel compelled to fill it.
    """
    active = library.active()
    if not active:
        return (
            "(библиотека приёмов пока пуста — в начале пути это нормально; "
            "записывай только то, о чём правда хочется вспомнить)"
        )
    lines: list[str] = []
    for e in active:
        head = f"- #{e.id} «{e.name}»"
        if e.created_in_series is not None:
            head += f" (из серии #{e.created_in_series})"
        lines.append(head)
        lines.append(f"    Когда доставать: {e.when_to_use}")
        # keep descriptions terse on the wire; the model has full access to
        # library.json on disk if it needed the raw dump, but we don't want
        # to bloat every prompt.
        desc = e.description.strip().replace("\n", " ")
        if len(desc) > 320:
            desc = desc[:317] + "..."
        lines.append(f"    Что это: {desc}")
        if e.code_snippet:
            lines.append("    (есть код-фрагмент)")
    return "\n".join(lines)


# Maximum chars we'll embed of a single snippet when building the step-1
# "expanded" library block. Long snippets get tail-truncated; model can
# work from the head + description.
_SNIPPET_MAX_CHARS = 1200


def _library_detailed_block(library: Library) -> str:
    """Full library dump for step 1: every active entry with its full
    description and, when available, its code snippet. This is the
    *reach-for-it* view — the model needs enough context to actually
    reuse an entry, not just know it exists.

    Returns empty string when the library is empty, so the caller can
    drop the whole section.
    """
    active = library.active()
    if not active:
        return ""
    parts: list[str] = []
    for e in active:
        origin = (
            f" (из серии #{e.created_in_series})"
            if e.created_in_series is not None
            else ""
        )
        parts.append(f"### #{e.id} — «{e.name}»{origin}")
        parts.append(f"**Когда доставать.** {e.when_to_use.strip()}")
        parts.append(f"**Что это.** {e.description.strip()}")
        if e.code_snippet:
            snippet = e.code_snippet.strip()
            if len(snippet) > _SNIPPET_MAX_CHARS:
                snippet = (
                    snippet[:_SNIPPET_MAX_CHARS]
                    + f"\n# ... (обрезано, полный код в library.json — "
                    f"{len(e.code_snippet)} симв.)"
                )
            parts.append("```python")
            parts.append(snippet)
            parts.append("```")
        parts.append("")  # blank line between entries
    return "\n".join(parts).rstrip()


def _full_memory(
    state: State, path_obj: dict | None, library: Library | None = None
) -> str:
    progress = _progress_block(state)
    progress_section = ""
    if progress:
        progress_section = f"## Где мы на пути\n\n{progress}\n\n"
    library_section = ""
    if library is not None:
        library_section = (
            "## Библиотека приёмов (копится между сериями)\n\n"
            f"{_library_memory_block(library)}\n\n"
        )
    return (
        "# Состояние памяти\n\n"
        f"{progress_section}"
        "## Серии\n\n"
        f"{_series_memory_block(state)}\n\n"
        "## artistic_path\n\n"
        f"{_path_memory_block(path_obj)}\n\n"
        f"{library_section}"
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
    library: Library


def _pick_series_survey_pngs(state: State, cur: Series, k: int) -> list[Path]:
    """Return up to k in-series reference images (first, middle picks) to
    show the model alongside the immediately previous image. Always excludes
    the very last iteration (which is handed to step 1 separately and will
    be the model's primary visual anchor).
    """
    if k <= 0 or cur is None or len(cur.iterations) < 2:
        return []
    iters = list(cur.iterations)[:-1]  # exclude most recent
    if not iters:
        return []
    # spread picks across the series: first, middle, etc.
    if k == 1:
        picks = [iters[0]]
    else:
        idxs = np.linspace(0, len(iters) - 1, num=k).round().astype(int)
        picks = [iters[int(i)] for i in idxs]
    # preserve order, de-dup
    seen: set[int] = set()
    out: list[Path] = []
    for n in picks:
        if n in seen:
            continue
        seen.add(n)
        p = OUTPUT_DIR / f"art_{n:03d}.png"
        if p.exists():
            out.append(p)
    return out


def _pick_past_series_closers(state: State, k: int) -> list[tuple[Series, Path]]:
    """Return the last rendered image of each closed series (up to k most
    recent closed series). Used to force cross-series visual differentiation
    on the first iteration of a new series.
    """
    if k <= 0:
        return []
    closed = [s for s in state.series if not s.is_open()]
    out: list[tuple[Series, Path]] = []
    for s in closed[-k:]:
        if not s.iterations:
            continue
        last = s.iterations[-1]
        p = OUTPUT_DIR / f"art_{last:03d}.png"
        if p.exists():
            out.append((s, p))
    return out


def _similarity_survey(state: State, window: int = SIMILARITY_WINDOW) -> str:
    """Return a short human-readable block about how visually similar the
    last `window` iterations are to each other. Blank string when there
    aren't enough datapoints yet.
    """
    nums = sorted(state.iterations.keys())[-window:]
    if len(nums) < 3:
        return ""
    fps: list[tuple[int, dict | None]] = []
    for n in nums:
        it = state.iterations.get(n)
        fp = it.get("fingerprint") if it else None
        fps.append((n, fp))
    # consecutive pairs
    rows: list[str] = []
    for (na, fa), (nb, fb) in zip(fps, fps[1:]):
        sim = similarity(fa, fb)
        rows.append(
            f"  - {na:03d} → {nb:03d}: sim={sim:.2f} ({similarity_bucket(sim)})"
        )
    # avg similarity to the latest work
    if fps[-1][1] is not None:
        avgs: list[float] = []
        for n, fp in fps[:-1]:
            avgs.append(similarity(fps[-1][1], fp))
        if avgs:
            rows.append(
                f"  - avg sim of {fps[-1][0]:03d} to previous "
                f"{len(avgs)} works: {sum(avgs) / len(avgs):.2f}"
            )
    return "\n".join(rows)


def step1_create(ctx: Ctx) -> None:
    log.info("step 1/6 — create art.py (iteration %03d)", ctx.n)
    prev_art = read_text(
        ART_SCRIPT, default="(первая итерация, предыдущего скрипта нет)"
    )
    prev_meta = (
        parse_art_py_metadata(prev_art)
        if prev_art
        else {
            "preserved": "",
            "mutated": "",
        }
    )
    cur = ctx.state.current_series()
    is_first_of_series = cur is not None and len(cur.iterations) == 0
    cur_block = (
        f"Текущая серия: #{cur.id} «{cur.name}» — {cur.thesis}"
        if cur
        else "Серии пока нет — эта работа откроет новую."
    )
    # cross-series contract block — only meaningful once the series has one
    contract_block = ""
    if cur and (cur.must_differ_from_previous_in or cur.forbidden_features):
        diff_line = (
            ", ".join(cur.must_differ_from_previous_in)
            if cur.must_differ_from_previous_in
            else "—"
        )
        forb_lines = (
            "\n".join(f"  - {f}" for f in cur.forbidden_features)
            if cur.forbidden_features
            else "  —"
        )
        contract_block = (
            "\n\n## Контракт серии (заявленный при открытии)\n\n"
            f"Должна отличаться от предыдущих серий по: **{diff_line}**.\n"
            f"Запрещённые признаки (не использовать в этой серии):\n{forb_lines}"
        )

    # Preserved/Mutated from the previous iteration — surfaced explicitly so
    # the model can't pretend it mutated something it kept. The rule: you
    # cannot mutate the *same* axis two iterations in a row; if you want to
    # keep changing it, you are in drift, not evolution.
    prov_block = ""
    if prev_meta["preserved"] or prev_meta["mutated"]:
        prov_block = (
            "\n\n## Родословная предыдущей работы\n\n"
            f"- Preserved было: {prev_meta['preserved'] or '—'}\n"
            f"- Mutated было: {prev_meta['mutated'] or '—'}\n"
            "В этой итерации мутируй **другую** ось, не ту же, что в прошлый раз. "
            "Иначе это не эволюция, а дрейф."
        )

    # Similarity survey — harness-computed, not model-self-reported.
    sim_block = _similarity_survey(ctx.state)
    sim_text = ""
    if sim_block:
        sim_text = (
            "\n\n## Визуальная схожесть последних работ (по перцептуальному "
            "хешу + цветовой гистограмме, 0 = разные, 1 = идентичные)\n"
            f"{sim_block}\n"
            f"Если последние работы держатся на sim ≥ {SIM_STRONG_REPEAT:.2f} — "
            "они фактически одинаковы; слова о мутации не считаются, нужно "
            "сменить палитру, композицию или алгоритмическое ядро."
        )

    # Detailed library block for step 1: the model needs the full
    # description + code snippet to actually reach for an entry, not just
    # know it exists. The compact view in _full_memory is for other steps.
    lib_detail_block = ""
    lib_detail = _library_detailed_block(ctx.library)
    if lib_detail:
        lib_detail_block = (
            "\n\n## Библиотека приёмов — полные записи\n\n"
            "Это твой собственный архив находок из прошлых серий. Используй "
            "его как художник использует свои альбомы: если какая-то запись "
            "по описанию `Когда доставать` точно отвечает задаче этой "
            "итерации — возьми её, вплети в работу, разверни её дальше. "
            "Совсем не обязательно доставать что-то каждый раз; но если "
            "поздняя фаза прогона, а ты ни разу не оперся на библиотеку — "
            "это повод задаться вопросом, зачем ты тогда её собирал.\n\n"
            "В комментарии `# Preserved:` прямо упомяни, какой записью (#id) "
            "ты воспользовался, если воспользовался. Не копируй код "
            "буквально — адаптируй под текущую серию, меняй параметры, "
            "соединяй с другими приёмами.\n\n"
            f"{lib_detail}"
        )

    # Build text section first, images second (vision payload).
    text = (
        f"{_full_memory(ctx.state, ctx.path_obj, ctx.library)}\n\n"
        f"## Предыдущий art.py\n\n"
        f"```python\n{prev_art[:12000]}\n```"
        f"{prov_block}"
        f"{contract_block}"
        f"{lib_detail_block}"
        f"{sim_text}\n\n"
        f"## Задача — Шаг 1 (итерация {ctx.n:03d})\n\n"
        f"{cur_block}\n\n"
        "Перед кодом ты увидишь изображения: предыдущую работу серии "
        "и несколько опорных кадров. Если серия только открывается — "
        "увидишь также последние кадры прошедших серий, от которых эта "
        "новая серия должна заметно отличаться.\n\n"
        f"Напиши полный `art.py` для этой итерации. "
        f"Скрипт должен сохранить `art_{ctx.n:03d}.png` {IMAGE_SIZE}×"
        f"{IMAGE_SIZE} в CWD. "
        "В начале файла — обязательные комментарии Iteration/Seed/Series/"
        "Preserved/Mutated. В Preserved перечисли, что удерживается от "
        "предыдущей работы (и если ты опираешься на запись библиотеки — "
        "укажи её как `library #<id>`); в Mutated — ровно ту ось, которую "
        "меняешь сейчас (одну-две, не все сразу).\n\n"
        "Ответь **только исходным кодом Python** — без JSON, без пояснений, "
        "без markdown-обёртки. Первая строка ответа должна быть первой "
        "строкой файла `art.py`."
    )

    content: list[dict] = [{"type": "text", "text": text}]

    # primary visual anchor: the previous rendered image
    prev_png: Path | None = None
    if ctx.state.iterations:
        prev_n = max(ctx.state.iterations)
        p = OUTPUT_DIR / f"art_{prev_n:03d}.png"
        if p.exists():
            prev_png = p
    if prev_png is not None:
        content.append(
            {
                "type": "text",
                "text": f"Предыдущая работа — art_{prev_png.stem[-3:]}.png:",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_downscaled(prev_png, size=512)},
            }
        )

    # in-series survey (excluding the last, which we just showed)
    if cur is not None and STEP1_SERIES_SURVEY > 0:
        survey = _pick_series_survey_pngs(ctx.state, cur, STEP1_SERIES_SURVEY)
        if survey:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Опорные кадры этой же серии (#{cur.id} «{cur.name}», "
                        "для удержания её ДНК):"
                    ),
                }
            )
            for p in survey:
                content.append({"type": "text", "text": f"  — {p.name}"})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_downscaled(p, size=384)},
                    }
                )

    # at series boundary: show last frames of past series so the new series
    # has to differ VISUALLY, not just verbally
    if is_first_of_series and STEP1_PAST_SERIES_SURVEY > 0:
        past = _pick_past_series_closers(ctx.state, STEP1_PAST_SERIES_SURVEY)
        if past:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Последние кадры предыдущих серий — эта новая работа "
                        "должна быть визуально отличима от каждого из них "
                        "(не только по смыслу):"
                    ),
                }
            )
            for s, p in past:
                content.append(
                    {
                        "type": "text",
                        "text": f"  — серия #{s.id} «{s.name}», {p.name}",
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_downscaled(p, size=384)},
                    }
                )

    art_py = llm_text(
        content,
        model=VISION_MODEL,
        temperature=0.95,
        max_tokens=12000,
        tag=f"step1_iter{ctx.n:03d}",
        expect_lang="python",
    )
    write_text(ART_SCRIPT, art_py)
    archive_art_script(ctx.n, art_py)
    log.info(
        "    wrote %s (%d bytes) + archived scripts/art_%03d.py",
        ART_SCRIPT.name,
        len(art_py),
        ctx.n,
    )


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
        archive_art_script(ctx.n, art_py)
        log.info("    applied fix attempt %d", attempt)

    raise RuntimeError("unreachable")


def step3_observe(ctx: Ctx, png: Path) -> dict:
    """Returns the validated structured observation.

    Shape matches SCHEMA_STEP3 — an enum-constrained visual audit plus a few
    free-text fields (palette, observation, what_works, what_fails). The
    result is persisted on the iteration record and the next iteration can
    diff its own audit against this one to detect drift / repetition.
    """
    log.info("step 3/6 — observe %s", png.name)
    art_src = read_text(ART_SCRIPT)[:4000]

    # Harness-computed similarity to the previous rendered image. We hand
    # the model the number so its own `similarity_to_previous` field is
    # informed, not guessed.
    sim_hint = ""
    prev_fp: dict | None = None
    prev_n: int | None = None
    if ctx.state.iterations:
        prev_n = max(ctx.state.iterations)
        prev_it = ctx.state.iterations.get(prev_n)
        prev_fp = prev_it.get("fingerprint") if prev_it else None
    if prev_fp is not None:
        this_fp = compute_fingerprint(png)
        sim = similarity(this_fp, prev_fp)
        sim_hint = (
            f"\n\nХарнесс измерил визуальную схожесть этой работы с "
            f"art_{prev_n:03d}.png: sim={sim:.2f} "
            f"({similarity_bucket(sim)}). Используй это как ориентир для "
            f"поля `similarity_to_previous`, но всё равно отвечай честно — "
            f"ты видишь образы, а числа лишь подсказка."
        )

    user = [
        {
            "type": "text",
            "text": (
                f"{_full_memory(ctx.state, ctx.path_obj, ctx.library)}\n\n"
                f"## Задача — Шаг 3 (итерация {ctx.n:03d})\n\n"
                "Посмотри на изображение и верни JSON строго по схеме "
                "Observation:\n"
                "- enum-поля (composition, shape_family, density, symmetry, "
                "element_count, negative_space, uses_blur_or_glow, "
                "similarity_to_previous) — короткие машиночитаемые ответы;\n"
                "- palette — 2–4 цвета с ролями (фон/доминант/акцент);\n"
                "- observation — 5–8 строк честного описания;\n"
                "- what_works / what_fails — по 1–2 предложения.\n"
                "Не выдумывай деталей, которых не видишь."
                f"{sim_hint}\n\n"
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
        temperature=0.5,
        max_tokens=2500,
        tag=f"step3_iter{ctx.n:03d}",
    )
    obs_text = (obj.get("observation") or "").strip()
    log.info(
        "    observation: comp=%s shape=%s density=%s sim=%s  | %s",
        obj.get("composition"),
        obj.get("shape_family"),
        obj.get("density"),
        obj.get("similarity_to_previous"),
        obs_text.splitlines()[0][:80] if obs_text else "(empty)",
    )
    return obj


def _format_observation_for_diary(obs: dict) -> str:
    """Render the structured step-3 observation into a compact block the
    step-4 prompt can lean on."""
    keys_enum = [
        ("composition", "composition"),
        ("shape_family", "shape_family"),
        ("density", "density"),
        ("symmetry", "symmetry"),
        ("element_count", "element_count"),
        ("negative_space", "negative_space"),
        ("uses_blur_or_glow", "uses_blur_or_glow"),
        ("similarity_to_previous", "similarity_to_previous"),
    ]
    lines: list[str] = []
    for k, label in keys_enum:
        if k in obs:
            lines.append(f"- {label}: {obs[k]}")
    if obs.get("palette"):
        lines.append(f"- palette: {obs['palette']}")
    if obs.get("observation"):
        lines.append(f"\n{obs['observation'].strip()}")
    if obs.get("what_works"):
        lines.append(f"\n+ {obs['what_works'].strip()}")
    if obs.get("what_fails"):
        lines.append(f"- {obs['what_fails'].strip()}")
    return "\n".join(lines)


def step4_diary(ctx: Ctx, observation: dict, measured_sim: float | None) -> dict:
    """Returns the validated diary entry (iteration-json shape).

    `observation` is the structured step-3 payload.
    `measured_sim` is the harness-computed similarity to the previous render
    (None on the very first iteration). We surface it as a hard number so
    the model's `repetition_check` field is grounded, not optimistic.
    """
    log.info("step 4/6 — diary entry")

    cur = ctx.state.current_series()
    cur_len = cur.length() if cur else 0

    obs_block = _format_observation_for_diary(observation)

    sim_block = ""
    if measured_sim is not None:
        sim_block = (
            f"\n\n## Измеренная визуальная схожесть с предыдущей работой\n\n"
            f"sim = {measured_sim:.2f} → bucket = "
            f"{similarity_bucket(measured_sim)}. "
            "Поле `repetition_check` в ответе обязано согласовываться с этим "
            "значением (отклонение на одну ступень — предел; иначе это "
            "самообман). В `repetition_note` назови ОДНУ конкретную ось, "
            "которая будет смещена в следующей итерации, если здесь "
            "mild/strong repeat."
        )

    # contract hint for the proposed_next_series at close-time
    past_series_block = ""
    closed = [s for s in ctx.state.series if not s.is_open()]
    if closed:
        past_series_block = (
            "\n\nПри close — в proposed_next_series поля "
            "must_differ_from_previous_in (минимум 2 оси) и "
            "forbidden_features (конкретные признаки, которые НЕ должны "
            "повториться в новой серии) должны быть содержательны и "
            "опираться на то, что уже было в прошлых сериях: "
            + "; ".join(f"#{s.id} «{s.name}»" for s in closed[-4:])
        )

    user = (
        f"{_full_memory(ctx.state, ctx.path_obj, ctx.library)}\n\n"
        f"## Наблюдение (итерация {ctx.n:03d}) — структурированно\n\n"
        f"{obs_block}"
        f"{sim_block}\n\n"
        "## Задача — Шаг 4\n\n"
        "Верни JSON по схеме DiaryEntry. Поле `series_status`:\n"
        "- 'continue' — работа продолжает текущую серию;\n"
        f"- 'close' — серия завершается этой работой. Разрешено только если "
        f"в серии уже ≥{MIN_SERIES_LEN} работ.\n"
        "При `close` обязательны содержательные `closed_reason` и "
        "`proposed_next_series` (со всеми полями, включая "
        "must_differ_from_previous_in и forbidden_features). Если текущей "
        "серии ещё нет (начало или принудительное закрытие предыдущей), "
        "`series_status` всё равно ставь 'continue' — харнесс сам откроет "
        "новую серию по твоему `proposed_next_series`, так что лучше "
        "заполни его и тут."
        f"{past_series_block}"
    )

    obj = llm_json(
        user,
        SCHEMA_STEP4,
        temperature=0.85,
        max_tokens=2500,
        tag=f"step4_iter{ctx.n:03d}",
    )

    # sanity: non-empty fields
    required = (
        "title",
        "what_i_see",
        "what_resonates",
        "what_is_silent",
        "where_next",
        "repetition_check",
        "repetition_note",
    )
    for k in required:
        if not obj.get(k) or len(obj[k].strip()) < 5:
            raise RuntimeError(
                f"diary field {k!r} is empty or too short: {obj.get(k)!r}"
            )

    # if harness measured strong repeat but the model reports novel, log a
    # warning (don't overwrite — the model sometimes has legit reasons, e.g.
    # intentional quote of a prior work). This shows up in run.log so you
    # can audit after the fact.
    if measured_sim is not None:
        bucket = similarity_bucket(measured_sim)
        if bucket == "strong_repeat" and obj["repetition_check"] == "novel":
            log.warning(
                "    model claims 'novel' but measured sim=%.2f is "
                "strong_repeat; leaving model's self-report but flagging.",
                measured_sim,
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

    if obj["series_status"] == "close":
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
    state: State, series: Series, path_obj: dict | None, library: Library
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
    past_closers = _pick_past_series_closers(state, k=10)
    # drop the just-closed series itself, if it sneaks in
    past_closers = [(s, p) for s, p in past_closers if s.id != series.id]

    # Mirror for the model: how often did *you* touch the library recently?
    # Helps counteract the "one addition per series" attractor we observed.
    recent_window = 5
    closed_so_far = [s for s in state.series if not s.is_open() and s.id != series.id]
    mirror_block = ""
    if closed_so_far:
        recent = closed_so_far[-recent_window:]
        touched = sum(
            1
            for s in recent
            if any(e.created_in_series == s.id for e in library.entries)
        )
        total_active = len(library.active())
        total_retired = len(library.entries) - total_active
        mirror_block = (
            "\n\n## Зеркало твоих привычек с библиотекой\n\n"
            f"За последние {len(recent)} закрытых серии ты пополнял "
            f"библиотеку в **{touched}** из них. В библиотеке сейчас "
            f"{total_active} активных записи и {total_retired} списанных. "
            "Если ты пополнял почти каждую серию — это не архив, а "
            "механическое ведение журнала; хорошая библиотека растёт "
            "скачками и подолгу стоит неизменной, пока твой язык "
            "действительно не сдвинется. `library_decision='nothing'` — "
            "это полноценный, честный ответ, а не отказ от работы."
        )

    header = (
        f"{_full_memory(state, path_obj, library)}\n\n"
        f"## Задача — Шаг 4b (Ретроспектива серии)\n\n"
        f"Серия #{series.id} «{series.name}» только что закрылась "
        f"на итерации {series.closed_at}. Её тезис был:\n"
        f"> {series.thesis}\n\n"
        f"Причина закрытия: {series.closed_reason or 'не указана'}.\n\n"
        f"Ниже — все {len(entries)} работ серии по порядку, "
        f"в уменьшенном виде (512px). Перед каждым изображением — "
        f"твоя же дневниковая запись о нём.\n\n"
        + (
            "После работ этой серии ты увидишь финальные кадры прошедших "
            "серий — чтобы судить эту серию не как изолированное "
            "произведение, а в контексте всего пути.\n\n"
            if past_closers
            else ""
        )
        + "Посмотри на серию как на единое произведение (и как на часть "
        "общего пути). Верни JSON SeriesRetrospective — поэтическое имя "
        f"всей серии (может совпадать с «{series.name}» или уточнять его), "
        "3–5 предложений об арке, что проявилось, что разочаровало, "
        "короткий вердикт, и мост к следующей серии.\n\n"
        "## Решение о библиотеке приёмов\n\n"
        "Схема требует от тебя сначала выбрать глагол в поле "
        "`library_decision`, а уже потом заполнять слоты. Варианты:\n\n"
        "- **`nothing`** — ничего не меняем в библиотеке. Это значение "
        "  по умолчанию и самый частый случай. Выбери его, если в этой "
        "  серии ты просто развивал ранее записанный приём, либо если "
        "  находка ещё не устоялась. Заполни `nothing_reason` одной "
        "  фразой, и оба массива оставь пустыми.\n"
        "- **`add`** — добавляем ровно одну (очень редко две) запись в "
        "  `library_additions`. Выбирай только если по тебе действительно "
        "  прошло что-то отдельное, к чему ты захочешь возвращаться в "
        "  будущих сериях. Если подобная запись уже есть — лучше "
        "  `add_and_retire` (добавить более точную, старую списать) или "
        "  просто `nothing`, чем плодить дубликат.\n"
        "- **`retire`** — списываем одну или несколько устаревших записей "
        "  через `library_retirements`, ничего не добавляя. Библиотека "
        "  должна сокращаться так же часто, как и расти.\n"
        "- **`add_and_retire`** — одновременная замена: обычно одна "
        "  запись в `library_additions` и одна в `library_retirements` "
        "  с `superseded_by: 0`.\n\n"
        "**Если в `library_additions` ты даёшь `code_snippet`, он должен "
        "быть рабочим.** Не заглушки вроде `def foo(): pass`, не `# "
        "simplified logic here`, не однострочные сигнатуры без тела. "
        f"Минимум ~{MIN_SNIPPET_CHARS} символов осмысленного кода. "
        "Если ты не готов написать настоящую реализацию — ставь "
        "`code_snippet: null` и оставь только словесное описание; это "
        "лучше, чем формально-правильный, но бесполезный стуб. Харнесс "
        "выбросит стуб и оставит только описание, и это будет "
        "зафиксировано в логах.\n\n"
        "Библиотека — подспорье, а не каркас. Она растёт скачками и "
        "подолгу не меняется. В ранних сериях её почти нет; в поздних "
        "ты реже пишешь в неё и чаще опираешься на уже собранное."
        f"{mirror_block}"
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

    # past-series closing frames, captioned briefly
    if past_closers:
        content.append(
            {
                "type": "text",
                "text": (
                    "— — —\n"
                    "Финальные кадры предыдущих серий "
                    "(для контекста и моста к следующей):"
                ),
            }
        )
        for s, p in past_closers:
            retro_title = (s.retrospective or {}).get("title") or s.name
            content.append(
                {
                    "type": "text",
                    "text": f"Серия #{s.id} «{s.name}» → «{retro_title}»",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_downscaled(p, size=384)},
                }
            )

    try:
        obj = llm_json(
            content,
            SCHEMA_RETRO,
            model=VISION_MODEL,
            temperature=0.7,
            max_tokens=4500,
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


def step5_path(ctx: Ctx, force: bool = False) -> dict | None:
    """Returns path_obj if rewritten, else None.

    By default path.json is only rewritten (a) the very first time, and
    (b) on a series boundary (caller passes force=True). Setting
    PATH_REWRITE_EVERY>0 in env re-enables periodic rewrites for debugging.
    The old per-6-iteration rewrite turned into an echo chamber where the
    model only ever tightened the current tendency.
    """
    first_time = ctx.path_obj is None
    periodic = PATH_REWRITE_EVERY > 0 and ctx.n > 1 and ctx.n % PATH_REWRITE_EVERY == 0
    if not (first_time or force or periodic):
        log.info("step 5/6 — path (skipped; only rewriting on boundary)")
        return None

    log.info(
        "step 5/6 — rewrite artistic_path (reason=%s)",
        "first" if first_time else ("boundary" if force else "periodic"),
    )
    is_first = ctx.n == 1 and first_time
    user = (
        f"{_full_memory(ctx.state, ctx.path_obj, ctx.library)}\n\n## Задача — Шаг 5\n\n"
        + (
            "Это итерация 001. Верни JSON ArtisticPath: поле `opening` "
            "заполни одной-двумя фразами, остальные поля заполни "
            "лаконичными набросками."
            if is_first
            else "Перепиши artistic_path целиком. Верни JSON ArtisticPath. "
            "Поле `opening` ставь null. Остальные семь полей — плотный срез "
            "(150–300 слов суммарно). Не бойся противоречить прошлому "
            "варианту: если выяснилось, что то, что тебя привлекало, на деле "
            "не работает — так и напиши."
        )
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
        if s.must_differ_from_previous_in or s.forbidden_features:
            if s.must_differ_from_previous_in:
                parts.append(
                    "*Обязана отличаться по:* "
                    + ", ".join(s.must_differ_from_previous_in)
                )
            if s.forbidden_features:
                parts.append(
                    "*Запрещено в этой серии:* " + "; ".join(s.forbidden_features)
                )
            parts.append("")
        for n in s.iterations:
            it = state.iterations.get(n)
            if it is None:
                continue
            parts.append(f"### Итерация {n:03d} — «{it['title']}»")
            parts.append(f"- **Что я вижу.** {it['what_i_see']}")
            parts.append(f"- **Что звучит.** {it['what_resonates']}")
            parts.append(f"- **Что молчит.** {it['what_is_silent']}")
            parts.append(f"- **Куда дальше.** {it['where_next']}")
            rep = it.get("repetition_check")
            if rep:
                rep_note = it.get("repetition_note") or ""
                sim = it.get("measured_similarity_to_previous")
                sim_str = f"{sim:.2f}" if isinstance(sim, (int, float)) else "—"
                parts.append(f"- **Повтор.** {rep} (sim={sim_str}) — {rep_note}")
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


def refresh_markdown(
    state: State, path_obj: dict | None, library: Library | None = None
) -> None:
    write_text(DIARY_PATH, render_diary(state))
    write_text(PATH_MD, render_path(path_obj))
    if library is not None:
        write_text(LIBRARY_MD, render_library(library))


# --------------------------------------------------------------------------- #
# main cycle
# --------------------------------------------------------------------------- #


def _ensure_open_series(
    state: State,
    at_iter: int,
    proposal: dict | None = None,
    *,
    library: Library | None = None,
) -> Series:
    """Open a new series at `at_iter` if none is currently open."""
    cur = state.current_series()
    if cur is not None:
        return cur
    if proposal and proposal.get("name") and proposal.get("thesis"):
        return state.open_series(
            proposal["name"],
            proposal["thesis"],
            at_iter,
            must_differ_from_previous_in=proposal.get("must_differ_from_previous_in"),
            forbidden_features=proposal.get("forbidden_features"),
        )
    # no proposal — ask the model in a tiny targeted call
    log.warning("    no open series and no proposal; asking model to name one")
    user = (
        f"{_full_memory(state, load_path_json(), library)}\n\n"
        "## Задача\n\nТекущей открытой серии нет, а харнессу она нужна. "
        "Предложи имя и тезис новой серии, которая осмысленно следует из "
        "прошлого пути. Также укажи:\n"
        "- must_differ_from_previous_in: массив ≥2 осей из "
        "[palette, composition, shape_family, density, symmetry, "
        "algorithmic_kernel, scale, texture], по которым эта серия "
        "обязана заметно отличаться от прошлых;\n"
        "- forbidden_features: список конкретных признаков, которые НЕ "
        "должны появляться в этой серии.\n"
        "Верни JSON с полями `name`, `thesis`, `must_differ_from_previous_in`, "
        "`forbidden_features`."
    )
    schema = {
        "name": "NewSeries",
        "strict": True,
        "schema": _obj(
            {
                "name": {"type": "string"},
                "thesis": {"type": "string"},
                "must_differ_from_previous_in": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "palette",
                            "composition",
                            "shape_family",
                            "density",
                            "symmetry",
                            "algorithmic_kernel",
                            "scale",
                            "texture",
                        ],
                    },
                    "minItems": 2,
                },
                "forbidden_features": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            }
        ),
    }
    obj = llm_json(
        user,
        schema,
        temperature=0.7,
        max_tokens=800,
        tag=f"open_series_iter{at_iter:03d}",
    )
    return state.open_series(
        obj["name"],
        obj["thesis"],
        at_iter,
        must_differ_from_previous_in=obj.get("must_differ_from_previous_in"),
        forbidden_features=obj.get("forbidden_features"),
    )


# Minimum length a meaningful snippet should have (after stripping). Below
# this we'd rather have no snippet than a one-liner signature.
MIN_SNIPPET_CHARS = 200

# Patterns that strongly indicate a stub instead of a real implementation.
# The test (in `_is_stub_snippet`) requires at least one of these **plus**
# a small actual-code budget before rejecting; a long real snippet that
# merely mentions "simplified" in a passing comment is fine.
_STUB_PATTERNS = (
    re.compile(r"\bpass\b\s*$", re.MULTILINE),
    re.compile(r"\.\.\.\s*$", re.MULTILINE),  # bare `...` ellipsis body
    re.compile(r"#\s*simplified", re.IGNORECASE),
    re.compile(r"#\s*conceptual", re.IGNORECASE),
    re.compile(r"#\s*logic\s+(to|for|here)", re.IGNORECASE),
    re.compile(r"#\s*(draw|compute|implement).*\s+here", re.IGNORECASE),
    re.compile(r"#\s*implementation\s+(of|for|goes)", re.IGNORECASE),
    re.compile(r"#\s*your\s+(code|logic)\s+here", re.IGNORECASE),
)


def _strip_code_content(src: str) -> str:
    """Return source with comments and blank lines stripped, so we can
    measure how much *actual* code there is."""
    out: list[str] = []
    for raw in src.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line)
    return "\n".join(out)


def _is_stub_snippet(snippet: str) -> tuple[bool, str]:
    """Decide whether `snippet` is a stub. Returns (is_stub, reason)."""
    s = snippet.strip()
    if not s:
        return True, "empty"
    if len(s) < MIN_SNIPPET_CHARS:
        return True, f"too short ({len(s)} < {MIN_SNIPPET_CHARS} chars)"
    code_only = _strip_code_content(s)
    if len(code_only) < MIN_SNIPPET_CHARS // 2:
        return True, f"mostly comments (code={len(code_only)} chars)"
    code_lines = [ln for ln in code_only.splitlines() if ln.strip()]
    # A real implementation almost always has 3+ non-comment lines.
    if len(code_lines) < 3:
        return True, f"only {len(code_lines)} non-comment code line(s)"
    for pat in _STUB_PATTERNS:
        m = pat.search(s)
        if m:
            return True, f"stub marker: {m.group(0)!r}"
    return False, ""


def _apply_library_changes(
    library: Library,
    retro: dict,
    *,
    series_id: int,
) -> None:
    """Honor library_additions / library_retirements from a retrospective.

    The model must first pick a `library_decision` enum value. The harness:
    - Drops additions/retirements that the chosen decision doesn't permit
      (so the model can't quietly 'nothing' + attach an addition anyway).
    - Rejects obviously thin additions and stubby code_snippets.
    - Logs 'nothing' with its reason for posterity.

    We tolerate the model setting `superseded_by: 0` as a sentinel meaning
    "the entry I just added in this same retrospective". With a single-
    addition retrospective that's unambiguous; with multiple we pick the
    first addition's id (rare and not worth more machinery).
    """
    decision = (retro.get("library_decision") or "").strip()
    nothing_reason = (retro.get("nothing_reason") or "").strip() or None
    raw_additions = retro.get("library_additions") or []
    raw_retirements = retro.get("library_retirements") or []

    if decision not in {"nothing", "add", "retire", "add_and_retire"}:
        log.warning(
            "    library_decision is missing/invalid (%r); treating as 'nothing'",
            decision,
        )
        decision = "nothing"

    # Honor the gate: drop slots the decision doesn't allow, and log loudly
    # when the model contradicted itself (decision=nothing but included an
    # addition, etc.). This is the main reason the enum exists.
    allow_add = decision in {"add", "add_and_retire"}
    allow_retire = decision in {"retire", "add_and_retire"}
    if raw_additions and not allow_add:
        log.warning(
            "    library_decision=%r but %d addition(s) attached; discarding",
            decision,
            len(raw_additions),
        )
        raw_additions = []
    if raw_retirements and not allow_retire:
        log.warning(
            "    library_decision=%r but %d retirement(s) attached; discarding",
            decision,
            len(raw_retirements),
        )
        raw_retirements = []

    if decision == "nothing":
        log.info(
            "    library: nothing (series #%d) — %s",
            series_id,
            nothing_reason or "reason not given",
        )
        return

    added_ids: list[int] = []
    for a in raw_additions:
        name = (a.get("name") or "").strip()
        desc = (a.get("description") or "").strip()
        when = (a.get("when_to_use") or "").strip()
        # refuse obviously empty additions so the library doesn't fill with
        # noise; ditto for absurdly short ones.
        if len(name) < 3 or len(desc) < 20 or len(when) < 5:
            log.warning(
                "    skipping library addition with too-thin content: "
                "name=%r when=%r desc_len=%d",
                name,
                when,
                len(desc),
            )
            continue
        # Sanity-check the code_snippet: if the model provided one, it has
        # to be a real implementation, not a `def foo(): pass` / `# logic
        # here` stub. If it fails the check, drop the snippet (keep the
        # written entry) and log — we prefer "description only" over "lie
        # that says it has code".
        snippet = a.get("code_snippet")
        if snippet:
            stub, why = _is_stub_snippet(snippet)
            if stub:
                log.warning(
                    "    library addition «%s»: code_snippet looks stubby "
                    "(%s); storing entry without snippet",
                    name,
                    why,
                )
                snippet = None
        entry = library.add(
            name=name,
            when_to_use=when,
            description=desc,
            code_snippet=snippet,
            series_id=series_id,
        )
        added_ids.append(entry.id)
        log.info(
            "    library: +#%d «%s» (from series #%d)%s",
            entry.id,
            entry.name,
            series_id,
            " [+code]" if entry.code_snippet else "",
        )

    for r in raw_retirements:
        eid = r.get("entry_id")
        if not isinstance(eid, int):
            continue
        superseded_raw = r.get("superseded_by")
        superseded: int | None = None
        if isinstance(superseded_raw, int):
            if superseded_raw == 0 and added_ids:
                superseded = added_ids[0]
            elif superseded_raw > 0 and library.by_id(superseded_raw) is not None:
                superseded = superseded_raw
        retired = library.retire(
            eid,
            reason=r.get("reason", ""),
            superseded_by=superseded,
        )
        if retired is None:
            log.warning(
                "    library retirement: id #%d not found or already retired", eid
            )
        else:
            log.info(
                "    library: -#%d «%s» retired%s",
                retired.id,
                retired.name,
                f" (superseded by #{superseded})" if superseded else "",
            )


def render_library(library: Library) -> str:
    """Render a human-readable view of the library for the exhibition dir."""
    if not library.entries:
        return ""
    lines = ["# Библиотека приёмов\n"]
    active = [e for e in library.entries if not e.retired]
    retired = [e for e in library.entries if e.retired]
    if active:
        lines.append("## Активные")
        lines.append("")
        for e in active:
            origin = (
                f" (из серии #{e.created_in_series})"
                if e.created_in_series is not None
                else ""
            )
            lines.append(f"### #{e.id} — {e.name}{origin}")
            lines.append(f"*Когда доставать:* {e.when_to_use}")
            lines.append("")
            lines.append(e.description.strip())
            lines.append("")
            if e.code_snippet:
                lines.append("```python")
                lines.append(e.code_snippet.strip())
                lines.append("```")
                lines.append("")
    if retired:
        lines.append("## Списаны")
        lines.append("")
        for e in retired:
            supp = f", заменена на #{e.superseded_by}" if e.superseded_by else ""
            lines.append(
                f"- #{e.id} «{e.name}» — "
                f"{e.retired_reason or 'причина не указана'}{supp}"
            )
    return "\n".join(lines).rstrip() + "\n"


def one_cycle(state: State, library: Library) -> None:
    n = state.next_iteration_number()
    path_obj = load_path_json()
    ctx = Ctx(n=n, state=state, path_obj=path_obj, library=library)
    log.info("=== iteration %03d ===", n)

    # ensure we have an open series BEFORE step 1 so its metadata can bake
    # into the art.py header. On iter 1 we don't have a proposal yet — open
    # a provisional series whose name/thesis comes from the path (if any) or
    # the model itself.
    if state.current_series() is None:
        _ensure_open_series(state, at_iter=n, library=library)

    step1_create(ctx)
    png = step2_render(ctx)

    # harness-side perceptual fingerprint: we need it for step3/step4 to
    # ground the similarity judgment in numbers, and we persist it on the
    # iter_json so future iterations can reuse it without re-reading PNGs.
    fingerprint = compute_fingerprint(png)

    # similarity to the immediately previous render (for step-4 grounding)
    measured_sim: float | None = None
    prev_fp: dict | None = None
    if state.iterations:
        prev_n = max(state.iterations)
        prev_it = state.iterations.get(prev_n)
        prev_fp = prev_it.get("fingerprint") if prev_it else None
        if prev_fp is not None:
            measured_sim = similarity(fingerprint, prev_fp)
            log.info(
                "    visual similarity to %03d: %.2f (%s)",
                prev_n,
                measured_sim,
                similarity_bucket(measured_sim),
            )

    observation = step3_observe(ctx, png)
    diary = step4_diary(ctx, observation, measured_sim)

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
        "repetition_check": diary.get("repetition_check"),
        "repetition_note": diary.get("repetition_note"),
        "series_status": diary["series_status"],
        "closed_reason": diary.get("closed_reason"),
        "proposed_next_series": diary.get("proposed_next_series"),
        "observation": observation,
        "measured_similarity_to_previous": measured_sim,
        "fingerprint": fingerprint,
        "image_path": png.name,
        "script_path": f"scripts/art_{n:03d}.py",
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
        retro = step4b_series_retrospective(state, closed_series_ref, path_obj, library)
        if retro is not None:
            state.set_series_retrospective(closed_series_ref.id, retro)
            _apply_library_changes(library, retro, series_id=closed_series_ref.id)

    # if closed, try to immediately open the next series using the proposal
    if closed_this_cycle:
        proposal = diary.get("proposed_next_series")
        # iter N is the last of the closed series; the NEW series opens at N+1
        _ensure_open_series(state, at_iter=n + 1, proposal=proposal, library=library)

    # step 5 — rewrite path. Force-rewrite on series boundary.
    new_path = step5_path(ctx, force=closed_this_cycle)
    if new_path is not None:
        path_obj = new_path

    # always refresh markdown views
    refresh_markdown(state, path_obj, library)
    log.info("=== iteration %03d done ===", n)


def main() -> None:
    log.info(
        "driver starting. model=%s vision=%s workdir=%s",
        MODEL,
        VISION_MODEL,
        OUTPUT_DIR,
    )
    state = State()
    library = Library()
    log.info(
        "loaded state: %d series, %d iterations, library: %d active / %d total",
        len(state.series),
        len(state.iterations),
        len(library.active()),
        len(library.entries),
    )
    while True:
        try:
            one_cycle(state, library)
        except KeyboardInterrupt:
            log.info("interrupted by user, bye")
            return
        except Exception:
            log.exception("cycle failed, sleeping 5s and continuing")
            time.sleep(5)


if __name__ == "__main__":
    main()
