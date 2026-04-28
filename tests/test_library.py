"""Unit tests for the technique-library curation + prompt templating.

Run with: uv run --with openai --with python-dotenv --with pillow --with numpy \
    python tests/test_library.py
"""

import os, sys, shutil, tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="image_gen_lib_test_"))
os.environ["OUTPUT_DIR"] = str(TMP)
os.environ["OPENAI_API_KEY"] = "test-dummy"
os.environ["EXPECTED_SERIES"] = "6"

# argparse will see tests/*.py's argv; strip to avoid --fresh leakage
sys.argv = [sys.argv[0]]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run  # noqa: E402


def ok(name, cond):
    mark = "OK " if cond else "FAIL"
    print(f"  {mark} {name}")
    if not cond:
        raise SystemExit(1)


def _mk_retro(
    decision="nothing",
    *,
    nothing_reason=None,
    additions=None,
    retirements=None,
):
    """Build a retrospective dict with the new schema shape."""
    return {
        "title": "t",
        "arc_summary": "a",
        "what_emerged": "e",
        "what_disappointed": "d",
        "verdict": "v",
        "bridge_to_next": "b",
        "library_decision": decision,
        "nothing_reason": nothing_reason,
        "library_additions": additions or [],
        "library_retirements": retirements or [],
    }


def _clean_library():
    if run.LIBRARY_JSON.exists():
        run.LIBRARY_JSON.unlink()


# -------- Library basics --------

lib = run.Library()
ok("empty library on fresh state", lib.entries == [])
ok("empty library.active() is []", lib.active() == [])

e1 = lib.add(
    name="Молочное свечение на охре",
    when_to_use="когда нужно смягчить геометрию без потери формы",
    description="Альфа-композит слоя с лёгким гауссом поверх твёрдой формы; keeps edges readable.",
    code_snippet="img = img.filter(ImageFilter.GaussianBlur(3))",
    series_id=1,
)
ok("first entry gets id 1", e1.id == 1)
ok("library has one active entry", len(lib.active()) == 1)

# persisted on disk
ok("library.json exists", run.LIBRARY_JSON.exists())

lib2 = run.Library()
ok("library roundtrips", len(lib2.entries) == 1 and lib2.entries[0].id == 1)
ok("code snippet preserved", "GaussianBlur" in lib2.entries[0].code_snippet)

# retire
retired = lib2.retire(1, reason="перестало звучать моим", superseded_by=None)
ok("retire returns entry", retired is not None and retired.retired)
ok("active() excludes retired", lib2.active() == [])
ok("retire idempotent-ish", lib2.retire(1, reason="again") is None)

# -------- _apply_library_changes: happy path + superseded_by:0 sentinel ----

_clean_library()
lib3 = run.Library()
e_old = lib3.add(
    name="Старая палитра охра/индиго",
    when_to_use="пыльные работы",
    description="двухцветная палитра с пыльным жёлтым и тёмным синим",
    code_snippet=None,
    series_id=1,
)

# A "real" code snippet that's long enough and doesn't look stubby.
REAL_SNIPPET = """import numpy as np
from PIL import Image

def apply_palette_shift(img_array, palette, intensity):
    h, w, _ = img_array.shape
    out = np.zeros_like(img_array, dtype=np.float32)
    for idx, (r, g, b) in enumerate(palette):
        weight = np.exp(-((img_array[..., 0] - r) ** 2) / (2 * 30.0 ** 2))
        out[..., 0] += weight * r * intensity
        out[..., 1] += weight * g * intensity
        out[..., 2] += weight * b * intensity
    return np.clip(out, 0, 255).astype(np.uint8)
"""

retro = _mk_retro(
    decision="add_and_retire",
    additions=[
        {
            "name": "Пыльная охра с индиговым акцентом",
            "when_to_use": "когда нужен тихий, приглушённый регистр",
            "description": "переформулировка старой записи с более точными ролями цветов и правилами композиции",
            "code_snippet": REAL_SNIPPET,
        }
    ],
    retirements=[{"entry_id": e_old.id, "reason": "слишком общее", "superseded_by": 0}],
)
run._apply_library_changes(lib3, retro, series_id=5)
ok("addition was applied", len(lib3.active()) == 1)
ok("old entry retired", lib3.by_id(e_old.id).retired is True)
new_entry = lib3.active()[0]
ok(
    "superseded_by rewritten to real new id",
    lib3.by_id(e_old.id).superseded_by == new_entry.id,
)
ok(
    "real snippet was kept",
    new_entry.code_snippet and "palette_shift" in new_entry.code_snippet,
)

# -------- too-thin additions are rejected ----------------------------------

_clean_library()
lib4 = run.Library()
retro_bad = _mk_retro(
    decision="add",
    additions=[
        {
            "name": "xx",
            "when_to_use": "ok",
            "description": "short",
            "code_snippet": None,
        },
        {
            "name": "valid name",
            "when_to_use": "ok when needed",
            "description": "this is sufficiently long to pass the sanity-check threshold",
            "code_snippet": None,
        },
    ],
)
run._apply_library_changes(lib4, retro_bad, series_id=2)
ok("thin entry rejected, valid kept", len(lib4.active()) == 1)
ok("kept entry is the longer one", lib4.active()[0].name == "valid name")

# -------- library_decision gating -------------------------------------------

# 1. decision=nothing + attached addition => addition discarded
_clean_library()
lib_gate = run.Library()
retro_lie = _mk_retro(
    decision="nothing",
    nothing_reason="just developing prior techniques",
    additions=[
        {
            "name": "Метод, который НЕ должен пройти",
            "when_to_use": "когда нужна иллюстрация того, как харнесс фильтрует ложь",
            "description": "Этот метод не должен попасть в библиотеку, потому что decision=nothing",
            "code_snippet": None,
        }
    ],
)
run._apply_library_changes(lib_gate, retro_lie, series_id=7)
ok("decision=nothing drops additions", len(lib_gate.active()) == 0)

# 2. decision=add + valid addition => added
_clean_library()
lib_add = run.Library()
retro_add = _mk_retro(
    decision="add",
    additions=[
        {
            "name": "Честное добавление",
            "when_to_use": "когда нужно показать, что add работает",
            "description": "это настоящая запись, которая должна попасть в библиотеку с decision=add",
            "code_snippet": None,
        }
    ],
)
run._apply_library_changes(lib_add, retro_add, series_id=8)
ok("decision=add accepts valid addition", len(lib_add.active()) == 1)

# 3. missing/unknown decision => treated as nothing
_clean_library()
lib_missing = run.Library()
bad = _mk_retro(
    additions=[
        {
            "name": "Запись без decision",
            "when_to_use": "когда decision вообще не задан",
            "description": "хитрый ответ без library_decision должен трактоваться как nothing",
            "code_snippet": None,
        }
    ],
)
del bad["library_decision"]
run._apply_library_changes(lib_missing, bad, series_id=9)
ok("missing library_decision treated as nothing", len(lib_missing.active()) == 0)

# 4. decision=retire + only retirement => retirement applied, no addition
_clean_library()
lib_ret = run.Library()
victim = lib_ret.add(
    name="Устаревшая запись",
    when_to_use="только для списания",
    description="эта запись создана специально, чтобы быть списанной в следующем шаге теста",
    code_snippet=None,
    series_id=1,
)
retro_ret = _mk_retro(
    decision="retire",
    retirements=[{"entry_id": victim.id, "reason": "устарело", "superseded_by": None}],
)
run._apply_library_changes(lib_ret, retro_ret, series_id=10)
ok("decision=retire retires the entry", lib_ret.by_id(victim.id).retired is True)

# -------- stub snippet detection --------------------------------------------

stub_cases = [
    ("def foo(): pass", "pass-only"),
    ("def foo():\n    ...", "ellipsis body"),
    (
        "def draw(canvas, center, radius):\n    # Simplified logic: draw a circle at center with given radius\n    pass",
        "simplified+pass",
    ),
    (
        "def helper():\n    # your code here\n    return None",
        "your code here",
    ),
    ("x = 1", "too short"),
    (
        "# This function computes a color palette\n# based on noise input\ndef palette():\n    return []",
        "mostly comments",
    ),
]
for snip, label in stub_cases:
    stub, why = run._is_stub_snippet(snip)
    ok(f"stub detected: {label}", stub)

non_stub = run._is_stub_snippet(REAL_SNIPPET)
ok("real snippet not stubby", non_stub[0] is False)

# long snippet with an incidental "simplified" word in a non-marker context still triggers
# (current matcher is conservative — that's fine); we verify the clean REAL_SNIPPET passes.

# -------- stubby code_snippet is stripped from the addition ----------------

_clean_library()
lib_stub = run.Library()
retro_stub = _mk_retro(
    decision="add",
    additions=[
        {
            "name": "Запись со заглушкой",
            "when_to_use": "проверка, что стуб-код отбрасывается",
            "description": "добавление с нормальным описанием, но стубовым код-фрагментом",
            "code_snippet": "def foo(): pass",
        }
    ],
)
run._apply_library_changes(lib_stub, retro_stub, series_id=11)
ok("entry kept despite stub snippet", len(lib_stub.active()) == 1)
ok("stub snippet was dropped", lib_stub.active()[0].code_snippet is None)

# -------- detailed library block for step 1 ---------------------------------

_clean_library()
lib_det = run.Library()
lib_det.add(
    name="Живой градиент",
    when_to_use="когда нужен медленный переход между двумя нотами",
    description="Линейное наслоение цветов с модуляцией альфы по синусу",
    code_snippet=REAL_SNIPPET,
    series_id=3,
)
lib_det.add(
    name="Сухая зернистость",
    when_to_use="когда нужна тактильность минеральной поверхности",
    description="Высокочастотный шум, отсекаемый по порогу, накладывается мультипликативно",
    code_snippet=None,  # description-only entry
    series_id=4,
)
detail = run._library_detailed_block(lib_det)
ok("detailed block has first entry name", "Живой градиент" in detail)
ok("detailed block has second entry name", "Сухая зернистость" in detail)
ok("detailed block embeds full description", "модуляцией альфы" in detail)
ok("detailed block contains the snippet body", "apply_palette_shift" in detail)
ok("detailed block uses code fences", "```python" in detail)
# empty library => empty block, so caller can skip the section
_clean_library()
ok(
    "empty library -> empty detailed block",
    run._library_detailed_block(run.Library()) == "",
)

# -------- progress phases via _progress_block ------------------------------

# We reuse the run.State from the main test; build a minimal one here.
st = run.State()
# no series yet — expected_series=6 but progress still shows position=1
pb = run._progress_block(st)
ok("progress block non-empty when EXPECTED_SERIES>0", bool(pb))
ok("progress mentions planned count", "~6" in pb)

# push closed count up to trigger late phase
for k in range(1, 6):
    s = st.open_series(f"s{k}", "th", at_iter=k)
    st.close_series(at_iter=k, reason="r")
# 5 closed, no open => completed=5, in_progress=0, position=5/6 ≈ 0.83 -> late
pb_late = run._progress_block(st)
ok("late phase shows 'ПОЗДНЯЯ'", "ПОЗДНЯЯ ФАЗА" in pb_late)

# EXPECTED_SERIES=0 disables block
saved = run.EXPECTED_SERIES
run.EXPECTED_SERIES = 0
ok("no progress block when disabled", run._progress_block(st) == "")
run.EXPECTED_SERIES = saved

# -------- prompt templating ------------------------------------------------

sp = run._build_system_prompt()
ok("prompt substitutes IMAGE_SIZE", f"{run.IMAGE_SIZE}×{run.IMAGE_SIZE}" in sp)
ok("prompt substitutes MIN_SERIES_LEN", str(run.MIN_SERIES_LEN) in sp)
ok("prompt substitutes EXPECTED_SERIES", "~6" in sp)
ok(
    "no unresolved {{TOKEN}} tokens in rendered prompt",
    "{{" not in sp or "}}" not in sp,
)
ok("prompt mentions library_decision", "library_decision" in sp)
ok(
    "prompt mentions stub snippets are rejected",
    "стуб" in sp.lower() or "заглушк" in sp.lower(),
)

# -------- library rendering -------------------------------------------------

md = run.render_library(lib3)
ok("rendered markdown has header", "# Библиотека приёмов" in md)
ok("rendered markdown has active section", "## Активные" in md)
ok("rendered markdown has retired section", "## Списаны" in md)

# -------- schema: retrospective has library fields required ----------------

p = run.SCHEMA_RETRO["schema"]
req = set(p["required"])
ok("library_additions required", "library_additions" in req)
ok("library_retirements required", "library_retirements" in req)
ok("library_decision required", "library_decision" in req)
ok("nothing_reason required", "nothing_reason" in req)
# decision enum has the four values
dec = p["properties"]["library_decision"]
ok(
    "library_decision enum is {nothing,add,retire,add_and_retire}",
    set(dec["enum"]) == {"nothing", "add", "retire", "add_and_retire"},
)

shutil.rmtree(TMP, ignore_errors=True)
print("\nall library tests passed")
