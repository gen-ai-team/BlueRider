"""Unit tests for run.py internals.

Run with: uv run --with openai --with python-dotenv python tests/test_state.py
"""
import os, sys, shutil, tempfile, json
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="image_gen_test_"))
os.environ["OUTPUT_DIR"] = str(TMP)
os.environ["OPENAI_API_KEY"] = "test-dummy"

# argparse will see tests/test_state.py's argv; strip to avoid --fresh leakage
sys.argv = [sys.argv[0]]

# ensure we import run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run  # noqa: E402


def ok(name, cond):
    mark = "OK " if cond else "FAIL"
    print(f"  {mark} {name}")
    if not cond:
        raise SystemExit(1)


# -------- State basics --------

st = run.State()
ok("empty state has no series", st.series == [])
ok("empty state has no iterations", st.iterations == {})
ok("next iter is 1 when empty", st.next_iteration_number() == 1)
ok("current_series is None when empty", st.current_series() is None)

# open series
s1 = st.open_series("Магматическая анатомия", "тезис1", at_iter=1)
ok("series id is 1", s1.id == 1)
ok("series has opened_at", s1.opened_at == 1)
ok("series is open", s1.is_open())
ok("current_series now returns s1", st.current_series() is s1)

# record iter 1 via iteration dict
for i in range(1, 8):
    it = {
        "iteration": i, "series_id": 1, "series_name": s1.name,
        "title": f"Работа {i}", "what_i_see": "...", "what_resonates": "...",
        "what_is_silent": "...", "where_next": "...",
        "series_status": "continue", "closed_reason": None,
        "proposed_next_series": None, "image_path": f"art_{i:03d}.png",
        "created_at": "2026-04-19T00:00:00+00:00",
    }
    st.save_iteration(it)
    st.append_to_current_series(i)

ok("series length 7 after 7 iterations", s1.length() == 7)
ok("next iter is 8", st.next_iteration_number() == 8)

# reload from disk: confirms persistence
st2 = run.State()
ok("reloaded series count", len(st2.series) == 1)
ok("reloaded iterations count", len(st2.iterations) == 7)
ok("reloaded series length", st2.current_series().length() == 7)

# close and open next
st2.close_series(at_iter=7, reason="Идея доведена до кульминации.")
ok("series is closed", not st2.current_series() is not None or
   st2.current_series() is None)  # None because it's closed
ok("current_series is None", st2.current_series() is None)
ok("last_closed_series returns it", st2.last_closed_series().name == "Магматическая анатомия")
ok("closed series has reason", st2.series[0].closed_reason == "Идея доведена до кульминации.")

s2 = st2.open_series("Центробежные разломы", "тезис2", at_iter=8)
ok("second series id is 2", s2.id == 2)
ok("current_series switched", st2.current_series().id == 2)

# -------- Series policy hints in memory block --------

# short series => MIN reminder
st3 = run.State()  # reload state with 2 series, s2 open, 0 iters in it
# adjust s2 to have 4 iters
for i in range(8, 12):
    it = {"iteration": i, "series_id": 2, "series_name": s2.name,
          "title": f"Работа {i}", "what_i_see": "...", "what_resonates": "...",
          "what_is_silent": "...", "where_next": "...",
          "series_status": "continue", "closed_reason": None,
          "proposed_next_series": None, "image_path": f"art_{i:03d}.png",
          "created_at": "2026-04-19T00:00:00+00:00"}
    st3.save_iteration(it)
    st3.append_to_current_series(i)

block = run._series_memory_block(st3)
ok("short series hint mentions MIN", "ещё нет" in block and "7 работ" in block)

# grow to SOFT limit
for i in range(12, 19 + 1):
    it = {"iteration": i, "series_id": 2, "series_name": s2.name,
          "title": f"Работа {i}", "what_i_see": "...", "what_resonates": "...",
          "what_is_silent": "...", "where_next": "...",
          "series_status": "continue", "closed_reason": None,
          "proposed_next_series": None, "image_path": f"art_{i:03d}.png",
          "created_at": "2026-04-19T00:00:00+00:00"}
    st3.save_iteration(it)
    st3.append_to_current_series(i)

# now s2 has 12 iters — still below SOFT
ok("series at 12 no limit hint", "МЯГКИЙ ЛИМИТ" not in run._series_memory_block(st3))

# push to 15
for i in range(20, 23):
    it = {"iteration": i, "series_id": 2, "series_name": s2.name,
          "title": "x", "what_i_see": "...", "what_resonates": "...",
          "what_is_silent": "...", "where_next": "...",
          "series_status": "continue", "closed_reason": None,
          "proposed_next_series": None, "image_path": f"art_{i:03d}.png",
          "created_at": "2026-04-19T00:00:00+00:00"}
    st3.save_iteration(it)
    st3.append_to_current_series(i)

ok("s2 has 15 iters", st3.current_series().length() == 15)
ok("series at SOFT lim triggers hint", "МЯГКИЙ ЛИМИТ" in run._series_memory_block(st3))

# push to 18 (HARD)
for i in range(23, 26):
    it = {"iteration": i, "series_id": 2, "series_name": s2.name,
          "title": "x", "what_i_see": "...", "what_resonates": "...",
          "what_is_silent": "...", "where_next": "...",
          "series_status": "continue", "closed_reason": None,
          "proposed_next_series": None, "image_path": f"art_{i:03d}.png",
          "created_at": "2026-04-19T00:00:00+00:00"}
    st3.save_iteration(it)
    st3.append_to_current_series(i)

ok("s2 has 18 iters", st3.current_series().length() == 18)
ok("series at HARD lim triggers hint", "ЖЁСТКИЙ ЛИМИТ" in run._series_memory_block(st3))


# -------- Renderers --------
diary_md = run.render_diary(st3)
ok("diary header present", "# Выставочный дневник" in diary_md)
ok("series header present", "## Серия 1 — «Магматическая анатомия»" in diary_md)
ok("closed marker present", "*(закрыта)*" in diary_md)
ok("iteration header present", "### Итерация 001 — «Работа 1»" in diary_md)
ok("separator present", "---" in diary_md)
ok("closed reason line present", "*Серия закрыта на итерации 7" in diary_md)

# render_path
path_obj = {
    "opening": None,
    "what_attracts_me": "цвет",
    "what_i_dislike": "серость",
    "what_works": "ритм",
    "where_i_stumble": "центр",
    "emerging_language": "гравитация",
    "current_series": "Разломы",
    "next_hypothesis": "паутина",
}
path_md = run.render_path(path_obj)
ok("path header", "# Путь художника" in path_md)
ok("section label", "## Что меня притягивает" in path_md)
ok("null opening omitted", "Начальная мысль" not in path_md)

# empty path
ok("empty path_obj -> empty render", run.render_path(None) == "")


# -------- Diary memory block --------
dm = run._diary_memory_block(st3, k=3)
ok("diary block has 3 latest", dm.count("Итерация") == 3)
ok("diary block uses series info", "Серия #2" in dm)


# -------- JSON schemas are wellformed --------
for schema in (run.SCHEMA_STEP1, run.SCHEMA_STEP3, run.SCHEMA_STEP4, run.SCHEMA_STEP5):
    ok(f"schema {schema['name']} has strict=True", schema["strict"] is True)
    ok(f"schema {schema['name']} has object root",
       schema["schema"]["type"] == "object")

# Step4 schema: series_status enum is correct
p = run.SCHEMA_STEP4["schema"]["properties"]
ok("series_status enum", p["series_status"]["enum"] == ["continue", "close"])


# -- series retrospective round-trip + rendering --------------------------

retro = {
    "title": "Магматическая анатомия (итог)",
    "arc_summary": "Семь работ прошли путь от пульсации к застыванию.",
    "what_emerged": "Зернистая плотность жара.",
    "what_disappointed": "Центр остался не решён.",
    "verdict": "Серия нашла тело, но не лицо.",
    "bridge_to_next": "Дальше — разрывы, центробежность.",
}
st3.set_series_retrospective(series_id=1, retro=retro)
ok("series 1 has retrospective", st3.series[0].retrospective is not None)
ok("retrospective round-trips via to_dict/from_dict",
   run.Series.from_dict(st3.series[0].to_dict()).retrospective == retro)

# reload from disk to confirm persistence
st4 = run.State()
ok("retrospective persisted", st4.series[0].retrospective == retro)

# diary rendering picks the rich footer when retrospective is set
diary_md2 = run.render_diary(st4)
ok("retro title appears in diary", "Взгляд назад — «Магматическая анатомия (итог)»" in diary_md2)
ok("retro arc_summary appears", "Семь работ прошли путь" in diary_md2)
ok("retro what_emerged appears", "Зернистая плотность жара" in diary_md2)
ok("retro what_disappointed appears", "Центр остался не решён" in diary_md2)
ok("retro verdict appears", "тело, но не лицо" in diary_md2)
ok("retro bridge appears", "разрывы, центробежность" in diary_md2)
ok("short closed_reason is NOT duplicated when retro is present",
   diary_md2.count("Серия закрыта на итерации 7") == 1)

# cleanup
shutil.rmtree(TMP, ignore_errors=True)
print("\nall tests passed")
