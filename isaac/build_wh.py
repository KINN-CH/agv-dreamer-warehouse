"""창고 자동 생성기 — seed마다 구조가 다른 물류창고 한 층 (Dreamer v5 학습·시험용, 깊이 BEV라 모양·배치가 다양해야 한다).
    python build_wh.py --seed 3 --out ~/agv/wh/s03      → warehouse.usda + env.json

seed가 정하는 것: 크기(44~64 × 44~60m), 랙 줄 방향(남북/동서), 랙 종류(선반·팔레트 랙·섞음), 통로 폭(선반 2.4~3.0m,
팔레트 3.0~4.0m), 중간 가로 통로 유무, 기둥 간격(10~14m), 입구 위치, 컨베이어·포장대 유무.
기본 틀(남북일 때): 남쪽 = 도크·입구·빈 바닥, 북쪽 = 랙 구역. 동서면 전체를 90° 돌린다.
처음 지도 = /World/env/fixed (벽·기둥·랙·선반·컨베이어·포장대·방호 기둥). 바닥 물건은 넣지 않는다 — 주행마다 물체 묶음에서 놓는다.
env.json: 입구(위치·방향), 통로(사각형·입구 점), 칸(통로마다 2.5m — 예 A3-07), 바닥 QR 표식 후보, 랙 줄 바닥 면적."""
import argparse
import json
import math
import random
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out).expanduser()
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(f"wh-{a.seed}")

W, D, H = rng.uniform(44, 64), rng.uniform(44, 60), rng.uniform(8.0, 12.0)
X0, X1, Y0, Y1 = -W / 2, W / 2, -D / 2, D / 2
NS = rng.random() < 0.5                                           # False면 전체를 90° 돌림 (랙 줄이 동서)
MODE = rng.choice(["shelf", "rack", "mixed", "mixed"])
AW_SHELF, AW_RACK = rng.uniform(2.4, 3.0), rng.uniform(3.0, 4.0)
YS0 = Y0 + D * rng.uniform(0.42, 0.55)                            # 랙 구역 남쪽 끝
YS1 = Y1 - rng.uniform(2.6, 3.6)                                  # 북쪽 벽과 사이 통로
CROSS = rng.random() < 0.5 and YS1 - YS0 > 18
SPACING = rng.uniform(10, 14)
CONVEYOR = rng.random() < 0.6


def T(x, y):                                                      # 기본 틀 → 월드
    return (x, y) if NS else (-y, x)


def TS(sx, sy):
    return (sx, sy) if NS else (sy, sx)


def TR(x0, y0, x1, y1):                                           # 사각형
    (ax, ay), (bx, by) = T(x0, y0), T(x1, y1)
    return [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]


stage = Usd.Stage.CreateNew(str(out / "warehouse.usda"))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
for g in ("fixed", "props", "marks", "lights"):
    UsdGeom.Xform.Define(stage, f"/World/{g}")
C = {"floor": (0.62, 0.63, 0.64), "wall": (0.86, 0.87, 0.88), "roof": (0.55, 0.56, 0.58), "column": (0.78, 0.78, 0.76),
     "guard": (0.95, 0.75, 0.05), "upright": (0.10, 0.30, 0.62), "beam": (0.93, 0.42, 0.06), "shelf": (0.55, 0.57, 0.60),
     "tote_y": (0.98, 0.80, 0.08), "tote_b": (0.12, 0.38, 0.75), "tote_g": (0.35, 0.36, 0.38), "pallet": (0.62, 0.47, 0.30),
     "box": (0.72, 0.56, 0.38), "box2": (0.80, 0.66, 0.48), "wrap": (0.85, 0.87, 0.88), "conv": (0.32, 0.33, 0.35),
     "belt": (0.08, 0.08, 0.08), "table": (0.80, 0.80, 0.78), "door": (0.50, 0.52, 0.55), "line_y": (0.98, 0.80, 0.05)}
MATS, N = {}, {"n": 0}


def mat(name):
    if name not in MATS:
        m = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/s")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*C[name]))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        MATS[name] = m
    return MATS[name]


def box(group, cx, cy, z0, sx, sy, sz, color, collide=True, name="b"):
    """기본 틀 좌표로 상자 하나 (월드로 돌려 놓는다)."""
    N["n"] += 1
    c = UsdGeom.Cube.Define(stage, f"/World/{group}/{name}_{N['n']}")
    c.CreateSizeAttr(1.0)
    (wx, wy), (wsx, wsy) = T(cx, cy), TS(sx, sy)
    xf = UsdGeom.XformCommonAPI(c)
    xf.SetTranslate(Gf.Vec3d(wx, wy, z0 + sz / 2))
    xf.SetScale(Gf.Vec3f(wsx, wsy, sz))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat(color))
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())


# ── 건물: 바닥(보이는 상자 + 충돌은 무한 평면), 벽, 지붕 ─────────────────
box("marks", 0, 0, -0.2, W + 2, D + 2, 0.2, "floor", collide=False, name="floor")
gp = UsdGeom.Plane.Define(stage, "/World/fixed/ground_collision")
gp.CreateAxisAttr("Z")
gp.CreatePurposeAttr("guide")
UsdPhysics.CollisionAPI.Apply(gp.GetPrim())
for cx, cy, sx, sy in ((0, Y0 - 0.15, W + 0.6, 0.3), (0, Y1 + 0.15, W + 0.6, 0.3), (X0 - 0.15, 0, 0.3, D), (X1 + 0.15, 0, 0.3, D)):
    box("fixed", cx, cy, 0.0, sx, sy, H, "wall", name="wall")
box("marks", 0, 0, H, W + 1, D + 1, 0.3, "roof", collide=False, name="roof")


# ── 랙 구역: 서쪽부터 줄·통로를 번갈아 ───────────────────────────────────
def shelf_row(xa, xb, ya, yb):
    units = max(1, round((yb - ya) / 1.2))
    unit, xm = (yb - ya) / units, (xa + xb) / 2
    for k in range(units + 1):
        for x in (xa + 0.02, xm, xb - 0.02):
            box("fixed", x, ya + k * unit, 0.0, 0.04, 0.04, 2.2, "shelf", name="shelf_post")
    for k in range(units):
        yc = ya + (k + 0.5) * unit
        for side in ((xa + xm) / 2, (xm + xb) / 2):
            for z in (0.12, 0.55, 1.0, 1.45, 1.9):
                box("fixed", side, yc, z - 0.03, 0.58, unit - 0.04, 0.03, "shelf", collide=z < 1.2, name="shelf_plate")
                for j in range(rng.choice([0, 1, 2, 2])):
                    box("props", side, yc + (j - 0.5) * 0.55, z, 0.55, 0.4, 0.3, rng.choice(["tote_y", "tote_y", "tote_b", "tote_g"]),
                        collide=z < 1.2, name="tote")


def rack_row(xa, xb, ya, yb):
    bays = max(1, round((yb - ya) / 2.7))
    bay = (yb - ya) / bays
    for face in (xa, xb):
        inner = face + 1.1 if face == xa else face - 1.1
        for k in range(bays + 1):
            for x in (face, inner):
                box("fixed", x, ya + k * bay, 0.0, 0.09, 0.09, 6.0, "upright", name="rack_upright")
        for k in range(bays):
            yc = ya + (k + 0.5) * bay
            for z in (1.6, 3.1, 4.6):
                for x in (face, inner):
                    box("fixed", x, yc, z - 0.12, 0.05, bay - 0.09, 0.12, "beam", collide=False, name="rack_beam")
            for z in (0.0, 1.6, 3.1, 4.6):
                for j in (-0.25, 0.25):
                    if rng.random() < 0.8:
                        lx, ly, hh = (face + inner) / 2, yc + j * bay, rng.uniform(0.7, 1.15)
                        box("props", lx, ly, z, 1.0, min(1.15, bay / 2 - 0.1), 0.14, "pallet", collide=z < 1.2, name="rack_pallet")
                        box("props", lx, ly, z + 0.14, 0.95, min(1.1, bay / 2 - 0.15), hh, rng.choice(["box", "box2", "wrap"]),
                            collide=z < 1.2, name="rack_load")


segs = [(YS0, YS1)]
if CROSS:
    ym, g = (YS0 + YS1) / 2, rng.uniform(3.0, 4.0)
    segs = [(YS0, ym - g / 2), (ym + g / 2, YS1)]
x, rows, k_row = X0 + rng.uniform(2.6, 4.0), [], 0
n_total = None
while True:
    kind = MODE if MODE != "mixed" else ("shelf" if (n_total is None or k_row < n_total // 2) else "rack")
    depth = 1.2 if kind == "shelf" else 2.3
    if x + depth > X1 - 2.6:
        break
    rows.append((kind, x, x + depth))
    aw = AW_SHELF if kind == "shelf" else AW_RACK
    x += depth + aw
    k_row += 1
    if n_total is None and MODE == "mixed":                       # 섞음: 줄 수를 먼저 세어 반반
        xx, n_total = X0 + 3.0, 0
        while xx + 2.3 <= X1 - 2.6:
            n_total += 1
            xx += 2.3 + (AW_SHELF + AW_RACK) / 2
for kind, xa, xb in rows:
    for ya, yb in segs:
        (shelf_row if kind == "shelf" else rack_row)(xa, xb, ya, yb)
aisles, slots, zones = [], {}, {}
for si, (ya, yb) in enumerate(segs):
    letter = "AB"[si]
    for i, ((_, _, xb), (_, xa2, _)) in enumerate(zip(rows, rows[1:])):
        name = f"{letter}{i + 1}"
        cx = (xb + xa2) / 2
        mouths = [T(cx, ya - 0.5)] + ([T(cx, yb + 0.5)])
        aisles.append({"name": name, "rect": TR(xb, ya, xa2, yb), "width": round(xa2 - xb, 2), "mouths": mouths,
                       "axis": "y" if NS else "x"})
        zones[name] = T(cx, (ya + yb) / 2)
        n_slot = max(1, int((yb - ya - 1.0) / 2.5))
        for k in range(n_slot):
            slots[f"{name}-{k + 1:02d}"] = T(cx, ya + 1.0 + 2.5 * k + 1.25)

# ── 빈 바닥: 기둥 격자, 컨베이어·포장대, 도크 ─────────────────────────────
ex = rng.uniform(X0 + 8, X1 - 8)
entrance = T(ex, Y0 + 4.0)
cols = []
for cx in [X0 + SPACING / 2 + k * SPACING for k in range(int(W // SPACING) + 1)]:
    for cy in [Y0 + 7 + k * SPACING for k in range(int((YS0 - Y0 - 9) // SPACING) + 1)]:
        if X0 + 2 < cx < X1 - 2 and cy < YS0 - 3 and math.dist((cx, cy), (ex, Y0 + 4)) > 4 and rng.random() < 0.85:
            box("fixed", cx, cy, 0.0, 0.6, 0.6, H, "column", name="column")
            box("fixed", cx, cy, 0.0, 0.8, 0.8, 1.0, "guard", name="column_guard")
            cols.append((cx, cy))
if CONVEYOR:
    c0, c1 = Y0 + 4.0, YS0 - 3.0
    box("fixed", X0 + 1.4, (c0 + c1) / 2, 0.0, 0.9, c1 - c0, 0.75, "conv", name="conveyor")
    box("fixed", X0 + 1.4, (c0 + c1) / 2, 0.75, 0.75, c1 - c0, 0.06, "belt", collide=False, name="conveyor_belt")
    yy = c0 + 1.0
    while yy < c1 - 1.0:
        box("fixed", X0 + 3.0, yy, 0.0, 0.9, 1.8, 0.9, "table", name="pack_table")
        yy += 4.0
dx = X0 + 4.0
while dx < X1 - 3:
    box("marks", dx, Y0 + 0.02, 0.0, 3.0, 0.04, 3.6, "door", collide=False, name="dock_door")
    if abs(dx - ex) > 3:
        for o in (-1.8, 1.8):
            box("fixed", dx + o, Y0 + 0.5, 0.0, 0.2, 0.2, 1.1, "guard", name="bollard")
    dx += 6.0
box("marks", 0, YS0 - 0.7, 0.0, W - 4, 0.12, 0.004, "line_y", collide=False, name="line")
k = 0
for lx in range(int(X0) + 4, int(X1), 8):
    for ly in range(int(Y0) + 4, int(Y1), 8):
        lt = UsdLux.RectLight.Define(stage, f"/World/lights/led_{k}")
        lt.CreateWidthAttr(2.4)
        lt.CreateHeightAttr(0.6)
        lt.CreateIntensityAttr(30000)
        wx, wy = T(lx, ly)
        UsdGeom.XformCommonAPI(lt).SetTranslate(Gf.Vec3d(wx, wy, H - 0.4))
        k += 1
UsdLux.DomeLight.Define(stage, "/World/lights/dome").CreateIntensityAttr(500)
stage.GetRootLayer().Save()

qr = [[*p, n] for n, p in slots.items()] + [[*m, f"{al['name']}-mouth{j}"] for al in aisles for j, m in enumerate(al["mouths"])]
qr += [[*T(gx, gy), f"G{gx:.0f}_{gy:.0f}"] for gx in range(int(X0) + 2, int(X1) - 1, 4) for gy in range(int(Y0) + 2, int(YS0) - 1, 4)]
env = {"usd": str(out / "warehouse.usda"), "seed": a.seed, "entrance": entrance, "entrance_yaw": math.pi / 2 if NS else math.pi,
       "zones": zones, "slots": slots, "aisles": aisles, "qr": qr,
       "routes": [["입구", z] for z in list(zones)[:6]],
       "rack_rows": [TR(xa, ya, xb, yb) for _, xa, xb in rows for ya, yb in segs], "props_root": "props",
       "bounds": TR(X0, Y0, X1, Y1), "params": {"W": round(W, 1), "D": round(D, 1), "NS": NS, "mode": MODE, "cross": CROSS,
                                                  "aisle_shelf": round(AW_SHELF, 2), "aisle_rack": round(AW_RACK, 2),
                                                  "conveyor": CONVEYOR, "columns": len(cols), "rows": len(rows)}}
(out / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=1))
print(f"seed {a.seed}: prims {N['n']} rows {len(rows)} aisles {len(aisles)} slots {len(slots)} qr {len(qr)} params {env['params']}")
