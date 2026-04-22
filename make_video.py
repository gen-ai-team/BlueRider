# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "moviepy>=2.1",
#     "opencv-python-headless>=4.8",
#     "imageio-ffmpeg>=0.4",
#     "numpy>=1.24",
#     "pillow>=10.0",
# ]
# ///
"""
make_video.py — promotional video generator for image_gen series.

For a chosen run folder (output_*), build a cinematic video in which every
iteration is shown exactly as it appears on the exhibition web page:

  - large image on the left, inside a dark frame with drop-shadow
  - title + series/iteration kicker overlaid on the image
  - four "thought" cards on the right:
        Что я вижу · Что звучит · Что молчит · Куда дальше
    (the last one tinted with the gold accent color, same as the web UI)

Between iterations images morph into each other using dense optical flow
(Farneback warp + alpha blend), while the side panel crossfades between
old and new thoughts.

Basic usage (default: one mp4 per series in output_gemini_2):

    uv run make_video.py
    uv run make_video.py output_gpt_2
    uv run make_video.py --series 3
    uv run make_video.py --series all --hold 6 --transition 1.2
    uv run make_video.py --iterations 10-30 --out videos/

Run `uv run make_video.py --help` for the full list of flags.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Palette & typography (mirrors web/style.css)
# ---------------------------------------------------------------------------

BG = (10, 10, 12)
BG_ELEV = (17, 17, 20)
BG_SOFT = (21, 21, 26)
LINE = (35, 35, 43)
LINE_SOFT = (27, 27, 34)
TEXT = (234, 234, 236)
TEXT_DIM = (167, 167, 176)
TEXT_MUTED = (107, 107, 118)
ACCENT = (231, 194, 122)
ACCENT_DIM = (140, 116, 70)
SERIES_BLUE = (159, 191, 255)

FONT_SERIF_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Palatino.ttc",
]
FONT_SANS_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]
FONT_SANS_BOLD_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]
FONT_MONO_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/Courier.ttc",
]


def _load_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Iteration:
    number: int
    series_id: int
    series_name: str
    series_index: int
    series_total: int
    title: str
    what_see: str
    what_resonates: str
    what_silent: str
    where_next: str
    image_path: Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_iterations(folder: Path) -> list[Iteration]:
    state_dir = folder / "state"
    series_json = state_dir / "series.json"
    iter_dir = state_dir / "iterations"
    if not series_json.is_file():
        raise SystemExit(f"not a run folder (missing {series_json})")
    if not iter_dir.is_dir():
        raise SystemExit(f"not a run folder (missing {iter_dir})")

    series_blob = _read_json(series_json).get("series", [])
    by_iter: dict[int, tuple[int, str, int, int]] = {}
    for s in series_blob:
        iters = list(s.get("iterations") or [])
        total = len(iters)
        for i, it in enumerate(iters, 1):
            by_iter[int(it)] = (int(s["id"]), str(s["name"]), i, total)

    out: list[Iteration] = []
    for p in sorted(iter_dir.glob("iter_*.json")):
        m = re.match(r"iter_(\d+)\.json$", p.name)
        if not m:
            continue
        n = int(m.group(1))
        blob = _read_json(p)
        img_rel = blob.get("image_path") or f"art_{n:03d}.png"
        img_abs = (folder / img_rel).resolve()
        if not img_abs.is_file():
            continue
        sid, sname, sidx, stotal = by_iter.get(
            n,
            (
                int(blob.get("series_id") or 0),
                str(blob.get("series_name") or "—"),
                0,
                0,
            ),
        )
        out.append(
            Iteration(
                number=n,
                series_id=sid,
                series_name=sname,
                series_index=sidx,
                series_total=stotal,
                title=str(blob.get("title") or "—"),
                what_see=str(blob.get("what_i_see") or "—"),
                what_resonates=str(blob.get("what_resonates") or "—"),
                what_silent=str(blob.get("what_is_silent") or "—"),
                where_next=str(blob.get("where_next") or "—"),
                image_path=img_abs,
            )
        )
    out.sort(key=lambda x: x.number)
    return out


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass
class Layout:
    W: int
    H: int
    pad: int
    gap: int
    img_box: tuple[int, int, int, int]
    side_box: tuple[int, int, int, int]


def build_layout(size: tuple[int, int]) -> Layout:
    W, H = size
    pad = max(24, int(min(W, H) * 0.024))
    gap = max(18, int(min(W, H) * 0.018))
    side_w = int(W * 0.36)
    side_x0 = W - pad - side_w
    img_x0, img_y0 = pad, pad
    img_x1 = side_x0 - gap
    img_y1 = H - pad
    return Layout(
        W=W,
        H=H,
        pad=pad,
        gap=gap,
        img_box=(img_x0, img_y0, img_x1, img_y1),
        side_box=(side_x0, pad, side_x0 + side_w, H - pad),
    )


# ---------------------------------------------------------------------------
# Pillow helpers
# ---------------------------------------------------------------------------


def _rounded_rect(draw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _fit_contain(img: Image.Image, box_w: int, box_h: int):
    iw, ih = img.size
    scale = min(box_w / iw, box_h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = img.resize((nw, nh), Image.LANCZOS)
    return resized, ((box_w - nw) // 2, (box_h - nh) // 2)


def _wrap_text(draw, text: str, font, max_w: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for w in words[1:]:
        trial = cur + " " + w
        b = draw.textbbox((0, 0), trial, font=font)
        if b[2] - b[0] <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _text_height(font) -> int:
    asc, desc = font.getmetrics()
    return asc + desc


def _truncate_to_width(draw, text: str, font, max_w: int) -> str:
    b = draw.textbbox((0, 0), text, font=font)
    if b[2] - b[0] <= max_w:
        return text
    ell = "…"
    lo, hi, best = 0, len(text), ""
    while lo < hi:
        mid = (lo + hi) // 2
        cand = text[:mid].rstrip() + ell
        bb = draw.textbbox((0, 0), cand, font=font)
        if bb[2] - bb[0] <= max_w:
            best = cand
            lo = mid + 1
        else:
            hi = mid
    return best or ell


def _draw_tracked(draw, xy, text, font, *, fill, tracking=0):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        b = draw.textbbox((0, 0), ch, font=font)
        x += (b[2] - b[0]) + tracking


# ---------------------------------------------------------------------------
# Image panel composer (split into base + caption so we can crossfade text
# independently during transitions)
# ---------------------------------------------------------------------------


def _render_image_panel_base(
    photo: Image.Image, box_w: int, box_h: int, *, zoom: float = 1.0
) -> Image.Image:
    radius = max(14, int(min(box_w, box_h) * 0.018))

    # Dark backdrop with a subtle radial highlight.
    panel = Image.new("RGB", (box_w, box_h), BG_ELEV)
    highlight = Image.new("RGB", (box_w, box_h), (26, 26, 32))
    mask = Image.new("L", (box_w, box_h), 0)
    r = int(max(box_w, box_h) * 0.55)
    cx, cy = box_w // 2, box_h // 2
    ImageDraw.Draw(mask).ellipse((cx - r, cy - r, cx + r, cy + r), fill=180)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, box_w // 8)))
    panel.paste(highlight, (0, 0), mask)

    # Optional Ken-Burns zoom + center crop.
    pw, ph = photo.size
    if zoom > 1.0:
        nw, nh = int(pw * zoom), int(ph * zoom)
        z = photo.resize((nw, nh), Image.LANCZOS)
        cx2, cy2 = nw // 2, nh // 2
        x0 = max(0, cx2 - pw // 2)
        y0 = max(0, cy2 - ph // 2)
        photo_z = z.crop((x0, y0, x0 + pw, y0 + ph))
    else:
        photo_z = photo

    inner_pad = int(min(box_w, box_h) * 0.04)
    fitted, (ox, oy) = _fit_contain(
        photo_z, box_w - 2 * inner_pad, box_h - 2 * inner_pad
    )
    panel.paste(fitted, (inner_pad + ox, inner_pad + oy))

    panel_rgba = panel.convert("RGBA")

    # Bottom shade for caption legibility.
    shade_h = int(box_h * 0.38)
    shade = Image.new("RGBA", (box_w, shade_h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    for i in range(shade_h):
        a = int(220 * (i / shade_h) ** 1.6)
        sd.line([(0, i), (box_w, i)], fill=(0, 0, 0, a))
    panel_rgba.alpha_composite(shade, (0, box_h - shade_h))

    # Rounded corners.
    corner = Image.new("L", (box_w, box_h), 0)
    ImageDraw.Draw(corner).rounded_rectangle(
        (0, 0, box_w, box_h), radius=radius, fill=255
    )
    panel_rgba.putalpha(corner)
    return panel_rgba


def _render_caption_overlay(
    box_w: int,
    box_h: int,
    *,
    series_name: str,
    series_idx: int,
    series_total: int,
    iter_num: int,
    title: str,
    alpha: float = 1.0,
) -> Image.Image:
    overlay = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    cap_px = int(box_w * 0.04)
    cap_py = int(box_h * 0.04)
    title_font = _load_font(FONT_SERIF_CANDIDATES, size=max(28, int(box_h * 0.058)))
    tag_font = _load_font(FONT_SANS_CANDIDATES, size=max(12, int(box_h * 0.018)))
    iter_font = _load_font(FONT_MONO_CANDIDATES, size=max(12, int(box_h * 0.018)))

    # Kicker position: near bottom, with title below it (matches web layout).
    kicker_y = box_h - cap_py - int(box_h * 0.11)

    series_txt = (series_name or "—").upper()
    series_txt = _truncate_to_width(d, series_txt, tag_font, int(box_w * 0.6))
    pad_px_x = int(tag_font.size * 0.7)
    pad_px_y = int(tag_font.size * 0.35)
    sb = d.textbbox((0, 0), series_txt, font=tag_font)
    tw, th = sb[2] - sb[0], sb[3] - sb[1]
    pill = (cap_px, kicker_y, cap_px + tw + 2 * pad_px_x, kicker_y + th + 2 * pad_px_y)
    _rounded_rect(d, pill, radius=9999, outline=(*SERIES_BLUE, 200), width=2)
    d.text(
        (cap_px + pad_px_x, kicker_y + pad_px_y - 1),
        series_txt,
        font=tag_font,
        fill=(*SERIES_BLUE, 255),
    )

    iter_txt = (
        f"СЕРИЯ {series_idx} / {series_total}  ·  #{iter_num:03d}"
        if series_total
        else f"#{iter_num:03d}"
    )
    d.text(
        (pill[2] + int(tag_font.size * 0.8), kicker_y + pad_px_y - 1),
        iter_txt,
        font=iter_font,
        fill=(*ACCENT, 255),
    )

    # Title below kicker.
    title_y = pill[3] + int(box_h * 0.015)
    max_title_w = box_w - 2 * cap_px
    title_lines = _wrap_text(d, title, title_font, max_title_w)[:2]
    line_h = int(title_font.size * 1.05)
    ty = title_y
    for ln in title_lines:
        d.text((cap_px, ty), ln, font=title_font, fill=(250, 250, 250, 255))
        ty += line_h

    if alpha < 1.0:
        a = overlay.split()[3]
        a = a.point(lambda v: int(v * alpha))
        overlay.putalpha(a)
    return overlay


def _blit_image_panel(
    base: Image.Image,
    panel_rgba: Image.Image,
    captions: list[Image.Image],
    layout: Layout,
) -> None:
    x0, y0, x1, y1 = layout.img_box
    box_w, box_h = x1 - x0, y1 - y0
    radius = max(14, int(min(box_w, box_h) * 0.018))

    merged = panel_rgba.copy()
    for cap in captions:
        merged.alpha_composite(cap)

    base_rgba = base.convert("RGBA")
    base_rgba.alpha_composite(merged, (x0, y0))
    base.paste(base_rgba.convert("RGB"))

    bd = ImageDraw.Draw(base)
    _rounded_rect(bd, (x0, y0, x1 - 1, y1 - 1), radius=radius, outline=LINE, width=1)


# ---------------------------------------------------------------------------
# Thought cards (right panel)
# ---------------------------------------------------------------------------


def _draw_thought_card(
    canvas: Image.Image,
    box,
    heading: str,
    body: str,
    *,
    accent: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    radius = 10

    card = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cd = ImageDraw.Draw(card)
    if accent:
        for i in range(h):
            t = i / max(1, h - 1)
            r = int(BG_SOFT[0] + (231 - BG_SOFT[0]) * 0.06 * (1 - t))
            g = int(BG_SOFT[1] + (194 - BG_SOFT[1]) * 0.06 * (1 - t))
            b = int(BG_SOFT[2] + (122 - BG_SOFT[2]) * 0.06 * (1 - t))
            cd.line([(0, i), (w, i)], fill=(r, g, b, 255))
        border = (231, 194, 122, int(255 * 0.35))
    else:
        cd.rectangle((0, 0, w, h), fill=(*BG_SOFT, 255))
        border = (*LINE_SOFT, 255)

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    card.putalpha(mask)

    bimg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(bimg).rounded_rectangle(
        (0, 0, w - 1, h - 1), radius=radius, outline=border, width=1
    )
    card.alpha_composite(bimg)

    pad = max(14, int(min(w, h) * 0.08))
    inner_w = w - 2 * pad
    heading_font = _load_font(FONT_SANS_BOLD_CANDIDATES, size=max(11, int(h * 0.10)))
    body_size = max(13, int(h * 0.115))
    body_font = _load_font(FONT_SERIF_CANDIDATES, size=body_size)

    td = ImageDraw.Draw(card)
    heading_col = ACCENT if accent else TEXT_MUTED
    _draw_tracked(
        td,
        (pad, pad),
        heading.upper(),
        heading_font,
        fill=(*heading_col, 255),
        tracking=2,
    )
    hh_h = _text_height(heading_font) + int(heading_font.size * 0.4)

    body_top = pad + hh_h
    lines = _wrap_text(td, body, body_font, inner_w)
    line_h = int(body_size * 1.38)
    max_lines = max(1, (h - body_top - pad) // line_h)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = _truncate_to_width(td, lines[-1] + " …", body_font, inner_w)
    ty = body_top
    for ln in lines:
        td.text((pad, ty), ln, font=body_font, fill=(*TEXT, 255))
        ty += line_h

    canvas.alpha_composite(card, (x0, y0))


def _render_side_panel(
    w: int, h: int, it: Iteration, *, alpha: float = 1.0
) -> Image.Image:
    """RGBA of the whole side panel (thoughts container)."""
    radius = 14
    panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(panel).rounded_rectangle(
        (0, 0, w, h),
        radius=radius,
        fill=(*BG_ELEV, 255),
        outline=(*LINE, 255),
        width=1,
    )

    pad = max(18, int(min(w, h) * 0.035))
    sec_font = _load_font(FONT_SERIF_CANDIDATES, size=max(18, int(h * 0.035)))
    sub_font = _load_font(FONT_SANS_CANDIDATES, size=max(11, int(h * 0.018)))
    pd = ImageDraw.Draw(panel)
    pd.text((pad, pad), "Мысли художника", font=sec_font, fill=(*TEXT, 255))
    sb = pd.textbbox((0, 0), "Мысли художника", font=sec_font)
    _draw_tracked(
        pd,
        (
            pad + (sb[2] - sb[0]) + int(sec_font.size * 0.5),
            pad + int(sec_font.size * 0.35),
        ),
        "О ТЕКУЩЕЙ РАБОТЕ",
        sub_font,
        fill=(*TEXT_MUTED, 255),
        tracking=1,
    )

    head_h = int(sec_font.size * 1.6) + pad
    cards_top = head_h
    cards_area_h = h - cards_top - pad
    gap = int(pad * 0.5)
    card_h = (cards_area_h - 3 * gap) // 4
    cx0, cx1 = pad, w - pad

    entries = [
        ("Что я вижу", it.what_see, False),
        ("Что звучит", it.what_resonates, False),
        ("Что молчит", it.what_silent, False),
        ("Куда дальше", it.where_next, True),
    ]
    y = cards_top
    for heading, body, accent in entries:
        _draw_thought_card(
            panel, (cx0, y, cx1, y + card_h), heading, body, accent=accent
        )
        y += card_h + gap

    if alpha < 1.0:
        a = panel.split()[3]
        a = a.point(lambda v: int(v * alpha))
        panel.putalpha(a)
    return panel


# ---------------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------------


def _blit_side_panel(
    base: Image.Image,
    panels: list[Image.Image],
    layout: Layout,
) -> None:
    x0, y0, x1, y1 = layout.side_box
    w, h = x1 - x0, y1 - y0
    merged = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for p in panels:
        merged.alpha_composite(p)
    base_rgba = base.convert("RGBA")
    base_rgba.alpha_composite(merged, (x0, y0))
    base.paste(base_rgba.convert("RGB"))


def render_hold_frame(
    layout: Layout, it: Iteration, photo: Image.Image, *, zoom: float = 1.0
) -> Image.Image:
    base = Image.new("RGB", (layout.W, layout.H), BG)

    ix0, iy0, ix1, iy1 = layout.img_box
    panel = _render_image_panel_base(photo, ix1 - ix0, iy1 - iy0, zoom=zoom)
    caption = _render_caption_overlay(
        ix1 - ix0,
        iy1 - iy0,
        series_name=it.series_name,
        series_idx=it.series_index,
        series_total=it.series_total,
        iter_num=it.number,
        title=it.title,
        alpha=1.0,
    )
    _blit_image_panel(base, panel, [caption], layout)

    sx0, sy0, sx1, sy1 = layout.side_box
    side = _render_side_panel(sx1 - sx0, sy1 - sy0, it, alpha=1.0)
    _blit_side_panel(base, [side], layout)
    return base


def render_transition_frame(
    layout: Layout,
    a: Iteration,
    b: Iteration,
    morphed_photo: Image.Image,
    t: float,
    *,
    zoom: float = 1.0,
) -> Image.Image:
    base = Image.new("RGB", (layout.W, layout.H), BG)

    ix0, iy0, ix1, iy1 = layout.img_box
    box_w, box_h = ix1 - ix0, iy1 - iy0
    panel = _render_image_panel_base(morphed_photo, box_w, box_h, zoom=zoom)
    captions: list[Image.Image] = []
    if t < 1.0:
        captions.append(
            _render_caption_overlay(
                box_w,
                box_h,
                series_name=a.series_name,
                series_idx=a.series_index,
                series_total=a.series_total,
                iter_num=a.number,
                title=a.title,
                alpha=1.0 - t,
            )
        )
    if t > 0.0:
        captions.append(
            _render_caption_overlay(
                box_w,
                box_h,
                series_name=b.series_name,
                series_idx=b.series_index,
                series_total=b.series_total,
                iter_num=b.number,
                title=b.title,
                alpha=t,
            )
        )
    _blit_image_panel(base, panel, captions, layout)

    sx0, sy0, sx1, sy1 = layout.side_box
    sw, sh = sx1 - sx0, sy1 - sy0
    side_layers: list[Image.Image] = []
    if t < 1.0:
        side_layers.append(_render_side_panel(sw, sh, a, alpha=1.0 - t))
    if t > 0.0:
        side_layers.append(_render_side_panel(sw, sh, b, alpha=t))
    _blit_side_panel(base, side_layers, layout)
    return base


# ---------------------------------------------------------------------------
# Optical-flow morph
# ---------------------------------------------------------------------------


def _to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return arr[:, :, ::-1]


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr[:, :, ::-1])


def compute_flows(a_bgr: np.ndarray, b_bgr: np.ndarray):
    a_gray = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    kwargs = dict(
        pyr_scale=0.5,
        levels=4,
        winsize=25,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    return (
        cv2.calcOpticalFlowFarneback(a_gray, b_gray, None, **kwargs),
        cv2.calcOpticalFlowFarneback(b_gray, a_gray, None, **kwargs),
    )


def warp_by_flow(img: np.ndarray, flow: np.ndarray, t: float) -> np.ndarray:
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xs + flow[..., 0] * t
    map_y = ys + flow[..., 1] * t
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def morph_blend(
    a_bgr: np.ndarray,
    b_bgr: np.ndarray,
    flow_ab: np.ndarray,
    flow_ba: np.ndarray,
    t: float,
) -> np.ndarray:
    wa = warp_by_flow(a_bgr, flow_ab, t)
    wb = warp_by_flow(b_bgr, flow_ba, 1.0 - t)
    return cv2.addWeighted(wa, 1.0 - t, wb, t, 0.0)


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------


def _prep_photo(path: Path, target: int = 1400) -> Image.Image:
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    scale = min(1.0, target / max(iw, ih))
    if scale < 1.0:
        im = im.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
    return im


def _same_canvas(photos: list[Image.Image]) -> list[Image.Image]:
    w = max(p.size[0] for p in photos)
    h = max(p.size[1] for p in photos)
    out = []
    for p in photos:
        c = Image.new("RGB", (w, h), (0, 0, 0))
        c.paste(p, ((w - p.size[0]) // 2, (h - p.size[1]) // 2))
        out.append(c)
    return out


def ease_in_out(t: float) -> float:
    return t * t * (3 - 2 * t)


def make_video(
    items: list[Iteration],
    out_path: Path,
    *,
    size: tuple[int, int] = (1920, 1080),
    fps: int = 30,
    hold: float = 7.0,
    transition: float = 1.5,
    ken_burns: bool = True,
    source_res: int = 1400,
) -> Path:
    if not items:
        raise SystemExit("no iterations to render")

    layout = build_layout(size)
    print(
        f"[video] {out_path.name}: {len(items)} iterations "
        f"{size[0]}x{size[1]} @ {fps}fps  "
        f"hold={hold}s  transition={transition}s",
        flush=True,
    )

    photos = _same_canvas(
        [_prep_photo(it.image_path, target=source_res) for it in items]
    )

    hold_frames = max(1, int(round(hold * fps)))
    trans_frames = max(1, int(round(transition * fps)))
    if len(items) == 1:
        total_frames = hold_frames
    else:
        total_frames = hold_frames * len(items) + trans_frames * (len(items) - 1)
    duration = total_frames / fps

    # Lazy cache of flow for the current pair only.
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def get_flows(i: int):
        if i not in cache:
            a_bgr = _to_cv(photos[i])
            b_bgr = _to_cv(photos[i + 1])
            fab, fba = compute_flows(a_bgr, b_bgr)
            cache.clear()
            cache[i] = (a_bgr, b_bgr, fab, fba)
        return cache[i]

    def state_of(fi: int):
        idx = 0
        rem = fi
        while idx < len(items):
            if rem < hold_frames:
                return ("hold", idx, rem)
            rem -= hold_frames
            if idx == len(items) - 1:
                return ("hold", idx, hold_frames - 1)
            if rem < trans_frames:
                return ("trans", idx, rem)
            rem -= trans_frames
            idx += 1
        return ("hold", len(items) - 1, hold_frames - 1)

    def make_frame(t_seconds: float) -> np.ndarray:
        fi = int(round(t_seconds * fps))
        fi = max(0, min(total_frames - 1, fi))
        state = state_of(fi)
        if state[0] == "hold":
            _, idx, sub = state
            it = items[idx]
            photo = photos[idx]
            z = 1.0 + (0.035 * (sub / max(1, hold_frames - 1)) if ken_burns else 0.0)
            img = render_hold_frame(layout, it, photo, zoom=z)
            return np.array(img)
        _, i, sub = state
        t = sub / max(1, trans_frames)
        te = ease_in_out(t)
        _, _, fab, fba = get_flows(i)
        a_bgr = _to_cv(photos[i])
        b_bgr = _to_cv(photos[i + 1])
        blended = morph_blend(a_bgr, b_bgr, fab, fba, te)
        photo = _to_pil(blended)
        zoom = 1.035 * (1 - te) + 1.0 * te if ken_burns else 1.0
        img = render_transition_frame(
            layout, items[i], items[i + 1], photo, te, zoom=zoom
        )
        return np.array(img)

    from moviepy import VideoClip

    clip = VideoClip(frame_function=make_frame, duration=duration)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(
        str(out_path),
        fps=fps,
        codec="libx264",
        audio=False,
        preset="medium",
        bitrate="6000k",
        threads=os.cpu_count() or 4,
        logger="bar",
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_size(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)x(\d+)", s.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError(f"bad --size {s!r} (expected WxH)")
    return int(m.group(1)), int(m.group(2))


def _parse_iter_range(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)-(\d+)", s)
    if not m:
        raise argparse.ArgumentTypeError(f"bad --iterations {s!r} (expected A-B)")
    a, b = int(m.group(1)), int(m.group(2))
    if a > b:
        raise argparse.ArgumentTypeError("--iterations: A must be <= B")
    return a, b


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\s/]+", "-", text)
    text = re.sub(r"[^\wа-яё\-]+", "", text, flags=re.UNICODE)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "series"


def select_jobs(
    all_items: list[Iteration],
    *,
    series: str | None,
    iter_range: tuple[int, int] | None,
) -> list[tuple[str, list[Iteration]]]:
    if iter_range is not None:
        a, b = iter_range
        sub = [it for it in all_items if a <= it.number <= b]
        if not sub:
            raise SystemExit(f"no iterations in range {a}-{b}")
        return [(f"iter_{a:03d}-{b:03d}", sub)]

    if series is None or series.lower() == "all":
        by_sid: dict[int, list[Iteration]] = {}
        for it in all_items:
            by_sid.setdefault(it.series_id, []).append(it)
        pairs = []
        for sid in sorted(by_sid):
            its = by_sid[sid]
            pairs.append((f"{sid:02d}_{_slugify(its[0].series_name)}", its))
        return pairs

    if series.lower() == "full":
        return [("full_run", all_items)]

    try:
        sid = int(series)
    except ValueError as e:
        raise SystemExit("--series must be a number, 'all' or 'full'") from e
    its = [it for it in all_items if it.series_id == sid]
    if not its:
        raise SystemExit(f"no iterations for series id {sid}")
    return [(f"{sid:02d}_{_slugify(its[0].series_name)}", its)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a promotional video for an image_gen run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "folder",
        nargs="?",
        default="output",
        help="Path to the run folder (must contain state/series.json).",
    )
    ap.add_argument(
        "--series",
        default="all",
        help="Series id, 'all' (one video per series), or 'full'.",
    )
    ap.add_argument(
        "--iterations",
        type=_parse_iter_range,
        default=None,
        help="Global iteration range A-B (overrides --series).",
    )
    ap.add_argument("--hold", type=float, default=7.0)
    ap.add_argument("--transition", type=float, default=1.5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=_parse_size, default=(1920, 1080), help="WxH")
    ap.add_argument("--out", type=Path, default=Path("videos"))
    ap.add_argument(
        "--no-kenburns",
        dest="ken_burns",
        action="store_false",
        default=True,
        help="Disable subtle zoom on static images.",
    )
    ap.add_argument(
        "--source-res",
        type=int,
        default=1400,
        help="Max source image edge used for morph (smaller = faster).",
    )
    args = ap.parse_args()

    # Make sure moviepy finds ffmpeg without brew install.
    try:
        import imageio_ffmpeg

        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"folder not found: {folder}")

    all_items = load_iterations(folder)
    if not all_items:
        raise SystemExit(f"no iterations found in {folder}")

    jobs = select_jobs(all_items, series=args.series, iter_range=args.iterations)
    out_dir = args.out.resolve()
    for label, items in jobs:
        out_path = out_dir / f"{folder.name}__{label}.mp4"
        make_video(
            items,
            out_path,
            size=args.size,
            fps=args.fps,
            hold=args.hold,
            transition=args.transition,
            ken_burns=args.ken_burns,
            source_res=args.source_res,
        )
        print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
