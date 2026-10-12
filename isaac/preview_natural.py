"""자연 장면(natural) 배치를 Isaac 없이 지도 위에 그려 확인 — logistics.py의 함수만 떼어 실행."""
import ast
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import nav  # noqa: E402

src = Path(__file__).with_name("logistics.py").read_text()
tree = ast.parse(src)
ns = {"math": math, "np": np, "nav": nav, "random": random}
for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("ENTRANCE", "ZONES", "TIMEOUT", "BOX_TYPES", "MAX_BOXES", "PERSON_HALF")
                                            for t in getattr(node.targets[0], "elts", [node.targets[0]])):
        exec(compile(ast.Module([node], []), "logistics", "exec"), ns)
    if isinstance(node, ast.FunctionDef) and node.name in ("sample_static", "along", "along_t", "natural"):
        exec(compile(ast.Module([node], []), "logistics", "exec"), ns)

M = Path("/data/ch/agv-wm/results/maps")
meta = json.loads((M / "inspect.json").read_text())
full = nav.GridMap(np.load(M / "grid.npy"), meta["cell"], meta["x0"], meta["y0"])
pool = {"forklift": [{"half": (1.745, 0.605)}], "SM_CardBoxA_01": [{"half": (0.35, 0.25)}],
        "SM_CardBoxB_01": [{"half": (0.25, 0.25)}], "SM_CardBoxD_01": [{"half": (0.19, 0.125)}]}
out = Path("/data/ch/agv-wm/results/natural_preview")
H, W = full.occ.shape if hasattr(full, "occ") else np.load(M / "grid.npy").shape
g = np.load(M / "grid.npy")
for zone in list(ns["ZONES"])[:6]:
    goal = ns["ZONES"][zone]
    path = nav.plan(full.coarse(0.1), ns["ENTRANCE"], goal, robot_r=0.7, soft_r=1.5)
    for seed in range(2):
        rng = random.Random(f"natural-{zone}-{seed}")
        try:
            acts = ns["natural"](rng, path, full, pool, ns["ENTRANCE"], goal)
        except RuntimeError as e:
            print(zone, seed, "건너뜀", e)
            continue
        gr = np.load(M / "grid_racks.npy")                          # AGV가 처음 아는 지도 = 랙·벽만
        rgb = np.full(g.shape + (3,), 235, np.uint8)
        rgb[g & ~gr] = (253, 186, 116)                               # 실제로는 있지만 AGV 지도에 없는 물건(가서 봄) — 연한 주황
        rgb[gr] = (60, 60, 60)
        im = Image.fromarray(rgb[::-1])
        dd = ImageDraw.Draw(im)
        px = lambda x, y: ((x - full.x0) / full.cell, H - (y - full.y0) / full.cell)   # noqa: E731
        racks = nav.GridMap(gr, meta["cell"], meta["x0"], meta["y0"])
        agv_path = nav.plan(racks.coarse(0.1), ns["ENTRANCE"], goal, robot_r=0.7, soft_r=1.5)   # AGV가 처음 짜는 길 (랙 지도)
        dd.line([px(*p) for p in agv_path], fill="#2563EB", width=6)
        cols = {"person": "#DC2626", "forklift": "#F59E0B"}
        info = []
        for a in acts:
            m = a.get("move")
            c = cols.get(a["kind"], "#8B5CF6")
            if m:
                dd.line([px(*p) for p in m["way"]], fill=c, width=3)
                T = sum(math.dist(p, q) / v for (p, q), v in zip(zip(m["way"], m["way"][1:]), m["speeds"])) + sum(m["waits"])
                info.append(f"{a['kind']} {len(m['way'])}점 {T:.0f}s")
            x, y = px(a["x"], a["y"])
            dd.ellipse([x - 9, y - 9, x + 9, y + 9], fill=c)
        im.crop((int(px(-30, 0)[0]), int(px(0, 25)[1]), int(px(10, 0)[0]), int(px(0, -22)[1]))).save(out / f"{zone}_s{seed}.png")
        print(zone, seed, info)
