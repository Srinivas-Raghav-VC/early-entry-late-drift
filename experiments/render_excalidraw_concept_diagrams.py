#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "Paper Template and Paper" / "Paper" / "figures"
SCENE_DIR = PROJECT_ROOT / "notes" / "excalidraw_scenes"

BASE_URL = os.environ.get("EXCALIDRAW_URL", "http://127.0.0.1:3010")

PALETTE = {
    "ink": "#1C1C1E", "muted": "#6E6E73",
    "orange": "#D55E00", "orange_fill": "#FFF2E8",
    "blue": "#0072B2", "blue_fill": "#EAF4FB",
    "green": "#009E73", "green_fill": "#EAF8F4",
    "magenta": "#CC79A7", "magenta_fill": "#FBEFF6",
    "gray": "#B8BCC5", "gray_fill": "#F5F6F8",
    "yellow": "#C58F00", "yellow_fill": "#FFF8E7",
    "red": "#C23B57", "red_fill": "#FCECF0",
}

def _post(path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return requests.post(f"{BASE_URL}{path}", json=payload or {}, timeout=120).json()

def _delete(path: str):
    requests.delete(f"{BASE_URL}{path}", timeout=30).raise_for_status()

def _get(path: str) -> Dict[str, Any]:
    return requests.get(f"{BASE_URL}{path}", timeout=30).json()

def text(el_id, x, y, value, font_size=20, color=None, width=None):
    # increased the width/height multipliers dramatically to fit the handwriting font
    return {
        "id": el_id, "type": "text", "x": x, "y": y,
        "width": width or max(120, int(len(max(value.splitlines(), key=len)) * font_size * 0.85)),
        "height": max(32, int(font_size * (2.2 + 1.2 * value.count("\n")))),
        "text": value, "fontSize": font_size, "fontFamily": 1,
        "strokeColor": color or PALETTE["ink"], "roughness": 1, "opacity": 100,
    }

def box(el_id, x, y, w, h, label=None, fill="#FFFFFF", stroke=None, stroke_width=1, roundness=16, font_size=20):
    out = {
        "id": el_id, "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
        "backgroundColor": fill, "strokeColor": stroke or PALETTE["ink"],
        "strokeWidth": stroke_width, "roughness": 1, "roundness": {"type": 3, "value": roundness},
        "opacity": 100,
    }
    if label is not None:
        out["label"] = {"text": label}
        out["fontSize"] = font_size
        out["fontFamily"] = 1
    return out

def ellipse(el_id, x, y, w, h, fill, stroke):
    return {
        "id": el_id, "type": "ellipse", "x": x, "y": y, "width": w, "height": h,
        "backgroundColor": fill, "strokeColor": stroke, "strokeWidth": 1, "roughness": 1,
    }

def arrow(el_id, x, y, points, stroke=None, stroke_width=1, start_arrow=None, end_arrow="arrow", elbowed=False):
    out = {
        "id": el_id, "type": "arrow", "x": x, "y": y, "points": points,
        "strokeColor": stroke or PALETTE["ink"], "strokeWidth": stroke_width,
        "roughness": 1, "opacity": 100,
    }
    if end_arrow: out["endArrowhead"] = end_arrow
    if start_arrow: out["startArrowhead"] = start_arrow
    if elbowed: out["elbowed"] = True
    return out

def _export_current_scene(stem: str):
    scene = _get("/api/elements")
    (SCENE_DIR / f"{stem}.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    svg_obj = _post("/api/export/image", {"format": "svg", "background": True})
    (FIG_DIR / f"{stem}.svg").write_text(svg_obj["data"], encoding="utf-8")
    png_obj = _post("/api/export/image", {"format": "png", "background": True})
    (FIG_DIR / f"{stem}.png").write_bytes(base64.b64decode(png_obj["data"]))

def _push_scene(elements, stem):
    _delete("/api/elements/clear")
    _post("/api/elements/batch", {"elements": elements})
    time.sleep(1.0)
    _export_current_scene(stem)

def make_timeline():
    e = []
    e.append(text("t1", 50, 30, "Where the main cells fail: a stage-local view", 28))
    e.append(text("t2", 50, 70, "The core distinction is entry versus continuation.", 16, PALETTE["muted"]))

    cols = [(210, "Early target entry"), (470, "Late continuation"), (730, "Whole-word regime")]
    for x, label in cols:
        e.append(box(f"c{x}", x, 110, 240, 50, label, PALETTE["gray_fill"], PALETTE["gray"], 18))

    rows = [
        ("1B Hindi", [("weak", PALETTE["red_fill"], PALETTE["red"]), ("weak", PALETTE["red_fill"], PALETTE["red"]), ("collapse", PALETTE["red_fill"], PALETTE["red"])]),
        ("1B Marathi", [("strong", PALETTE["green_fill"], PALETTE["green"]), ("weak", PALETTE["yellow_fill"], PALETTE["yellow"]), ("same-script\nhybrid", PALETTE["yellow_fill"], PALETTE["yellow"])]),
        ("1B Telugu", [("mostly good", PALETTE["green_fill"], PALETTE["green"]), ("weak", PALETTE["red_fill"], PALETTE["red"]), ("bank\nretrieval", PALETTE["yellow_fill"], PALETTE["yellow"])]),
        ("1B Bengali", [("weak", PALETTE["red_fill"], PALETTE["red"]), ("weak", PALETTE["red_fill"], PALETTE["red"]), ("catastrophic\nweak", PALETTE["red_fill"], PALETTE["red"])]),
        ("4B Telugu", [("strong", PALETTE["green_fill"], PALETTE["green"]), ("strong", PALETTE["green_fill"], PALETTE["green"]), ("stable\ncomposition", PALETTE["green_fill"], PALETTE["green"])]),
    ]
    for i, (label, cells) in enumerate(rows):
        y = 180 + i*75
        e.append(text(f"r{i}", 60, y+15, label, 20))
        for j, (txt, fill, stroke) in enumerate(cells):
            # width increased from 210 to 240
            e.append(box(f"b{i}_{j}", cols[j][0], y, 240, 65, txt, fill, stroke, 18))

    return e

def make_arch():
    e = []
    e.append(text("t1", 50, 30, "Hybrid Gemma 3 stack and the late sites studied", 28))
    e.append(text("t2", 50, 70, "Schematic of the hybrid local/global architecture and localized bottlenecks.", 16, PALETTE["muted"]))

    def draw_lane(y_start, label, widths, labels, fills, strokes):
        e.append(text(f"l{y_start}", 15, y_start + 25, label, 18))
        x = 185
        for i, (w, lab, fill, stroke) in enumerate(zip(widths, labels, fills, strokes)):
            # font_size=14 so "global\nrefresh" (7-char max line) fits in 90px boxes
            e.append(box(f"lb{y_start}_{i}", x, y_start, w, 82, lab, fill, stroke, 14))
            if i < len(widths) - 1:
                e.append(arrow(f"la{y_start}_{i}", x + w + 2, y_start + 41, [[0, 0], [12, 0]], PALETTE["gray"]))
            x += w + 13

    # 1B: regular blocks 90px, global-refresh 95px, study sites 145px
    widths1  = [82, 82, 95, 82, 82, 95, 148, 148]
    fills1   = [PALETTE["gray_fill"], PALETTE["gray_fill"], PALETTE["blue_fill"],
                PALETTE["gray_fill"], PALETTE["gray_fill"], PALETTE["blue_fill"],
                PALETTE["orange_fill"], PALETTE["blue_fill"]]
    strokes1 = [PALETTE["gray"], PALETTE["gray"], PALETTE["blue"],
                PALETTE["gray"], PALETTE["gray"], PALETTE["blue"],
                PALETTE["orange"], PALETTE["blue"]]
    labels1  = ["early", "mid", "global\nrefresh", "mid", "late", "global\nrefresh",
                "L25 MLP\nHindi", "L26 layer\noutput\nTelugu"]
    draw_lane(125, "Gemma 3 1B", widths1, labels1, fills1, strokes1)

    # 4B: same widths but one extra "late" block
    widths2  = [82, 82, 95, 82, 82, 95, 82, 148, 148]
    fills2   = [PALETTE["gray_fill"], PALETTE["gray_fill"], PALETTE["blue_fill"],
                PALETTE["gray_fill"], PALETTE["gray_fill"], PALETTE["blue_fill"],
                PALETTE["gray_fill"], PALETTE["blue_fill"], PALETTE["blue_fill"]]
    strokes2 = [PALETTE["gray"], PALETTE["gray"], PALETTE["blue"],
                PALETTE["gray"], PALETTE["gray"], PALETTE["blue"],
                PALETTE["gray"], PALETTE["blue"], PALETTE["blue"]]
    labels2  = ["early", "mid", "global\nrefresh", "mid", "late", "global\nrefresh",
                "late", "L30\u2013L32\nwriters", "L34 layer\noutput\nTelugu"]
    draw_lane(248, "Gemma 3 4B", widths2, labels2, fills2, strokes2)

    return e

def make_axes():
    e = []
    e.append(text("t1", 50, 30, "Two-axis regime map of multilingual ICL", 28))
    e.append(text("t2", 50, 65, "Same-script support helps early entry; scale stabilizes late continuation.", 16, PALETTE["muted"]))

    # Three regime zones — bg3 ends at x=940 so 4B labels have breathing room
    e.append(box("bg1",  80, 125, 230, 285, fill=PALETTE["red_fill"],    stroke=PALETTE["red"]))
    e.append(box("bg2", 320, 125, 270, 285, fill=PALETTE["yellow_fill"], stroke=PALETTE["yellow"]))
    e.append(box("bg3", 600, 125, 340, 285, fill=PALETTE["green_fill"],  stroke=PALETTE["green"]))

    e.append(text("tx1",  98, 145, "collapse", 18, PALETTE["red"]))
    e.append(text("tx2", 336, 145, "entry without\nstable continuation", 16, PALETTE["yellow"]))
    e.append(text("tx3", 618, 145, "stable composition", 18, PALETTE["green"]))

    # L-shaped axis: corner at (80, 425), going up 305 then right 880
    e.append(arrow("axes_L", 70, 420, [[0, -305], [0, 0], [880, 0]],
                   stroke=PALETTE["ink"], start_arrow="arrow", end_arrow="arrow", elbowed=True))
    e.append(text("xlbl", 690, 435, "late query-specific continuation", 15))
    e.append(text("ylbl",   8, 160, "early\ntarget\nentry", 16))

    # Points: (lang, 1B_x, 1B_y, 4B_x, 4B_y, lbl1B_dx_dy, lbl4B_dx_dy)
    # 1B points stay in left two zones; 4B points inside bg3 (600-940)
    points = [
        ("Hindi",   178, 335, 710, 185, (-38,  22), ( 14, -20)),
        ("Marathi", 355, 225, 675, 200, (-50,  22), (-72,  18)),
        ("Bengali", 142, 375, 640, 260, (-38,  22), ( 14,   0)),
        ("Tamil",   375, 275, 705, 225, (-38,  22), ( 14,   0)),
        ("Telugu",  405, 285, 815, 195, (-28,  22), ( 14,  -6)),
    ]
    for i, (lang, x1, y1, x2, y2, l1, l2) in enumerate(points):
        e.append(arrow(f"tr_{i}", x1+10, y1+10,
                       [[0, 0], [x2-x1-20, y2-y1-20]], PALETTE["gray"]))
        e.append(ellipse(f"p1_{i}", x1, y1, 20, 20,
                         fill=PALETTE["orange_fill"], stroke=PALETTE["orange"]))
        e.append(ellipse(f"p4_{i}", x2, y2, 20, 20,
                         fill=PALETTE["blue_fill"], stroke=PALETTE["blue"]))
        e.append(text(f"lt1_{i}", x1+l1[0], y1+l1[1], f"1B {lang}", 14))
        e.append(text(f"lt2_{i}", x2+l2[0], y2+l2[1], f"4B {lang}", 14))

    return e

def make_bridge():
    e = []
    e.append(text("t1", 50, 30, "Why Hindi yields a simple patch but Telugu does not", 28))
    e.append(text("t2", 50, 70, "The key difference is where the failure sits in the computation.", 16, PALETTE["muted"]))

    # width increased 400 -> 440
    e.append(box("hbg", 50, 120, 440, 480, fill=PALETTE["orange_fill"], stroke=PALETTE["orange"]))
    e.append(box("tbg", 510, 120, 440, 480, fill=PALETTE["blue_fill"], stroke=PALETTE["blue"]))
    e.append(text("ht", 70, 140, "Hindi branch (Gemma 3 1B)", 22))
    e.append(text("tt", 530, 140, "Telugu branch (Gemma 3 1B/4B)", 22))

    y = 190
    for i, (htxt, ttxt) in enumerate([
        ("Failure stage:\nearly target-entry competition", "Failure stage:\nlate bank-attractor drift"),
        ("Implementational bottleneck:\nlast-token L25 MLP", "Implementational bottleneck:\nL26 / L34 residual site"),
        ("Compact harmful subspace:\ncore candidate [5486, 2299]", "Mechanistic picture:\nmediated carried continuation state"),
        ("Practical result:\nfixed patch improves held-out CER", "Practical result:\nstatic late-state patches are weaker")
    ]):
        e.append(box(f"hb_{i}", 70, y, 400, 80, htxt, "#FFF", PALETTE["orange"], 18))
        e.append(box(f"tb_{i}", 530, y, 400, 80, ttxt, "#FFF", PALETTE["blue"], 18))
        if i < 3:
            e.append(arrow(f"ha_{i}", 270, y+80, [[0,0], [0,30]], PALETTE["orange"]))
            e.append(arrow(f"ta_{i}", 730, y+80, [[0,0], [0,30]], PALETTE["blue"]))
        y += 110

    # width increased 670 -> 720, height increased 80 -> 100
    e.append(box("bottom_strip", 155, 620, 720, 100, fill=PALETTE["gray_fill"], stroke=PALETTE["gray"]))
    e.append(text("bottom_text", 175, 640, "Mechanistically real does not mean equally steerable:\nHindi looks like a compact signed bottleneck, while Telugu\nlooks like a later and more conditional state bottleneck.", 18, width=680))
    return e

def main():
    try:
        requests.get(f"{BASE_URL}/health", timeout=10)
    except:
        return
    _push_scene(make_timeline(), "fig_stage_timeline_excalidraw")
    _push_scene(make_arch(), "fig_architecture_sites_excalidraw")
    _push_scene(make_axes(), "fig_axes_excalidraw")
    _push_scene(make_bridge(), "fig_mechanism_bridge_excalidraw")

if __name__ == "__main__":
    main()
