"""logistics.py 결과를 격자 지도 위에 그린다.

    python plot_nav.py <logistics 폴더>          # run.json → nav_topdown.png (회색 = 계획, 색 = 실제 궤적)
    python plot_nav.py <boxes 폴더> <격자 폴더>   # boxes.json → boxes_sheet.png (장면마다 상자·대응 없음·재계획 궤적)
    python plot_nav.py <hazards/종류 폴더> <격자 폴더>   # results.json → sheet.png (주황 = 놓인 물체, 보라 선 = 움직인 길)
        merged.json(A·B·B′·C)이 있으면 그것으로 — 빨강 A 대응 없음, 파랑 B 규칙, 초록 C Dreamer
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

d = Path(sys.argv[1])
gd = Path(sys.argv[2]) if len(sys.argv) > 2 else d
g, m = np.load(gd / "grid.npy"), json.loads((gd / "inspect.json").read_text())
H = g.shape[0]
px = lambda x, y: ((x - m["x0"]) / m["cell"], H - (y - m["y0"]) / m["cell"])   # noqa: E731
base = Image.fromarray(np.where(g[::-1], 60, 255).astype(np.uint8)).convert("RGB")
crop = (0, int(px(0, 31.5)[1]), base.width, int(px(0, -24.5)[1]))


def ok(r):
    return r.get("arrived") and not r.get("contact")


if any((d / f).exists() for f in ("boxes.json", "results.json", "merged.json")):
    old = (d / "boxes.json").exists()
    src = "boxes.json" if old else ("merged.json" if (d / "merged.json").exists() else "results.json")
    scens = json.loads((d / src).read_text())
    resp = "replan" if old else "respond"
    draw = [("none", "#e6194b"), (resp, "#4363d8" if src == "merged.json" else "#2e8b57")]
    if src == "merged.json":
        draw.append(("wm", "#2e8b57"))
    tiles = []
    for s in scens:
        img = base.copy()
        dr = ImageDraw.Draw(img)
        dr.line([px(*q) for q in s[resp]["plan0"]], fill="#bbbbbb", width=9)
        for b in s["boxes" if old else "actors"]:
            if "move" in b:
                mv = b["move"]
                dr.line([px(*w) for w in mv.get("way", [mv["a"], mv["b"]])], fill="#8e44ad", width=5)
                bx, by = px(*mv["b"])
                dr.ellipse([bx - 9, by - 9, bx + 9, by + 9], fill="#8e44ad")
            else:
                dr.polygon([px(*c) for c in b["corners"]], fill="#d2691e")
        for cond, col in draw:
            t = s[cond]["traj"]
            dr.line([px(*q) for q in t], fill=col, width=6)
            ex, ey = px(*t[-1])
            dr.ellipse([ex - 12, ey - 12, ex + 12, ey + 12], fill=col if ok(s[cond]) else "black")
        xs = [q[0] for q in s[resp]["plan0"]]
        cx = (min(xs) + max(xs)) / 2                                # 경로 가운데로 가로 20m × 세로 49m 잘라 1m = 14px
        x0, y0 = px(cx - 10, 29.5)
        x1, y1 = px(cx + 10, -19.5)
        tile = img.crop((int(x0), int(y0), int(x1), int(y1))).resize((280, 686))
        td = ImageDraw.Draw(tile)
        td.rectangle([0, 0, 280, 30], fill="white")
        td.text((4, 2), f"{s['route'][12:]} s{s['seed']}  n={s.get('n_boxes', s.get('n'))}", fill="black")
        if src == "merged.json":
            td.text((4, 16), "  ".join(f"{n} {'OK' if ok(s[c]) else 'HIT'}" for n, c in (("A", "none"), ("B", resp), ("C", "wm"))),
                    fill="black")
        else:
            td.text((4, 16), f"none {'OK' if ok(s['none']) else 'HIT'}  {resp} {'OK' if ok(s[resp]) else 'FAIL'}"
                             f" {s[resp]['time_s']}s", fill="black")
        tiles.append(tile)
    cols = max(1, max(s["seed"] for s in scens) + 1)
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (284 * cols, 690 * rows), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * 284, (k // cols) * 690))
    sheet.save(d / ("boxes_sheet.png" if old else ("sheet_ABC.png" if src == "merged.json" else "sheet.png")))
else:
    runs = json.loads((d / "run.json").read_text())
    img = base.copy()
    dr = ImageDraw.Draw(img)
    cols = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#000075"]
    for r, c in zip(runs, cols):
        if "plan" not in r:
            continue
        dr.line([px(*q) for q in r["plan"]], fill="#bbbbbb", width=9)
        dr.line([px(*q) for q in r["traj"]], fill=c, width=4)
        ex, ey = px(*r["traj"][-1])
        dr.ellipse([ex - 7, ey - 7, ex + 7, ey + 7], fill=c if r["arrived"] else "black")
        sx, sy = px(*r["plan"][0])
        dr.rectangle([sx - 8, sy - 8, sx + 8, sy + 8], outline=c, width=3)
    for k, r in enumerate(runs):
        dr.text((40, 40 + 18 * k), f"{r['route']}: {'OK' if r['arrived'] else 'FAIL'} {r.get('time_s', '')}s "
                f"clr {r.get('min_body_clearance', '')}m", fill=cols[k % len(cols)])
    img.crop(crop).save(d / "nav_topdown.png")
