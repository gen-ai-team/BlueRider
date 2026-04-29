# BlueRider

`BlueRider` - это эксперимент с автономным генеративным искусством. LLM пишет `art.py`, рендерит изображение, анализирует результат, объединяет работы в серии и ведет структурированную историю своего художественного пути.

### Основные части

- `run.py` - основной цикл генерации, хранения состояния и управления сериями.
- `prompt.md` - системный промпт, который задает художественный процесс.
- `web.py` + `web/` - небольшой веб-интерфейс для просмотра прогонов как выставки.
- `make_video.py` - сборка промо-видео по готовому прогону.

### Быстрый старт

1. Создайте `.env` на основе `.env.example` и укажите `OPENAI_API_KEY`.
2. Убедитесь, что интерпретатор для сгенерированного `art.py` содержит `pillow`, `numpy`, `scipy` и `noise`. Его можно задать через `ART_PYTHON`.
3. Запустите генератор:

```bash
uv run run.py
```

4. Откройте веб-интерфейс выставки:

```bash
uv run web.py
```

5. При необходимости соберите видео по папке прогона:

```bash
uv run make_video.py output
```

### Результаты

По умолчанию результаты пишутся в `./output`. Там находятся изображения, актуальный `art.py`, лог запуска и структурированное состояние.


## English

`BlueRider` is an autonomous generative art experiment. An LLM writes `art.py`, renders an image, reflects on the result, groups works into series, and keeps a structured history of its artistic path.

### Main parts

- `run.py` - the main loop that generates images, stores state, and manages series.
- `prompt.md` - the system prompt that defines the artistic process.
- `web.py` + `web/` - a small exhibition UI for browsing runs.
- `make_video.py` - builds promo videos from a finished run.

### Quick start

1. Create `.env` from `.env.example` and set `OPENAI_API_KEY`.
2. Make sure the interpreter used for generated `art.py` has `pillow`, `numpy`, `scipy`, and `noise` installed. You can point to it with `ART_PYTHON`.
3. Start the generator:

```bash
uv run run.py
```

4. Open the exhibition UI:

```bash
uv run web.py
```

5. Optionally build a video from a run folder:

```bash
uv run make_video.py output
```

### Output

The default output folder is `./output`. It contains generated images, the latest `art.py`, run logs, and structured state.