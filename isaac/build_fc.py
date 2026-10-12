"""쿠팡·아마존식 풀필먼트 센터(FC) 한 층을 USD로 만든다 — 두 번째 시험 창고.
참고: 아마존 FC 사진 투어(노란 토트, 광택 콘크리트 바닥, 바닥 표시, 컨베이어), 쿠팡 대구 FC 르포(다단 선반 집품, 토트 → 컨베이어,
층별 보관). 상표·로고는 넣지 않는다.

    python build_fc.py --out ~/agv/fc        → fc_warehouse.usda + fc_env.json

좌표는 첫 창고(Simple_Warehouse)와 같은 틀 — 입구 (−10.4, −18) 북쪽을 봄, 구역 6곳은 통로 가운데 y = 17,
랙 줄 아래 끝 y = 9 — 그래서 돌발 장면 생성(sample_mover·sample_mover2·natural)을 그대로 쓴다.
구성 (x −34~14, y −22~27, 천장 10m)
  서쪽 P-1~P-3: 회색 철제 다단 선반(2.2m, 5단, 양면 1.2m) + 토트 — 낱개 집품
  동쪽 R-1~R-3: 파란 기둥·주황 빔 팔레트 랙(6m, 바닥+3단, 양면 2.3m) + 팔레트 적재물 — 보관
  남쪽 입고장: 도크 문 6개 + 노란 방호 기둥, 팔레트·박스 적치 / 서쪽 벽: 컨베이어 + 토트 + 포장대 4개
  기둥 6개(12m 간격), 노란 통로선·정지선, 초록 보행자 통로, 천장 LED 격자.
처음 지도(AGV가 아는 것) = /World/env/fixed (벽·기둥·랙·선반·컨베이어·포장대·방호 기둥).
/World/env/props (팔레트·박스·대차·세워 둔 지게차) = 지도에 없음 → 가서 본다. 랙에 실린 짐도 props지만 랙 바닥 면적은 지도가 막는다."""
import argparse
import json
import random
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="~/agv/fc")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
out = Path(a.out).expanduser()
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(a.seed)
ASSETS = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1"

X0, X1, Y0, Y1, H = -34.0, 14.0, -22.0, 27.0, 10.0
ROW_Y0, ROW_Y1 = 9.0, 25.0                                    # 랙·선반 줄의 아래·위 끝
SHELF_ROWS = [(-31.0, -29.8), (-27.2, -26.0), (-23.4, -22.2), (-19.6, -18.4)]     # 양면 선반 줄 x 범위
RACK_ROWS = [(-16.4, -14.1), (-10.7, -8.4), (-5.0, -2.7), (0.7, 3.0)]            # 양면 팔레트 랙 줄
ZONES = {"P-1": (-28.5, 17.0), "P-2": (-24.7, 17.0), "P-3": (-20.9, 17.0),
         "R-1": (-12.4, 17.0), "R-2": (-6.7, 17.0), "R-3": (-1.0, 17.0)}
ENTRANCE = (-10.4, -18.0)
COLUMNS = [(x, y) for x in (-22.0, -10.0, 2.0) for y in (-9.0, 1.0)]
DOORS = [-28.0, -22.0, -16.0, -4.0, 2.0, 8.0]

stage = Usd.Stage.CreateNew(str(out / "fc_warehouse.usda"))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(root.GetPrim())
for g in ("fixed", "props", "marks", "lights"):
    UsdGeom.Xform.Define(stage, f"/World/{g}")

MATS = {}


def mat(name, rgb, rough=0.6, metal=0.0):
    if name not in MATS:
        m = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/s")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        MATS[name] = m
    return MATS[name]


C = {"floor": ((0.62, 0.63, 0.64), 0.3), "wall": ((0.86, 0.87, 0.88), 0.8), "roof": ((0.55, 0.56, 0.58), 0.9),
     "column": ((0.78, 0.78, 0.76), 0.7), "guard": ((0.95, 0.75, 0.05), 0.5), "upright": ((0.10, 0.30, 0.62), 0.45),
     "beam": ((0.93, 0.42, 0.06), 0.45), "shelf": ((0.55, 0.57, 0.60), 0.4), "yellow_tote": ((0.98, 0.80, 0.08), 0.5),
     "blue_tote": ((0.12, 0.38, 0.75), 0.5), "gray_tote": ((0.35, 0.36, 0.38), 0.6), "pallet": ((0.62, 0.47, 0.30), 0.85),
     "box": ((0.72, 0.56, 0.38), 0.85), "box2": ((0.80, 0.66, 0.48), 0.85), "wrap": ((0.85, 0.87, 0.88), 0.25),
     "conv": ((0.32, 0.33, 0.35), 0.5), "belt": ((0.08, 0.08, 0.08), 0.8), "table": ((0.80, 0.80, 0.78), 0.6),
     "screen": ((0.05, 0.05, 0.06), 0.2), "door": ((0.50, 0.52, 0.55), 0.6), "line_y": ((0.98, 0.80, 0.05), 0.5),
     "line_g": ((0.10, 0.62, 0.30), 0.5), "line_w": ((0.95, 0.95, 0.95), 0.5), "cage": ((0.62, 0.64, 0.66), 0.4),
     "charger": ((0.15, 0.55, 0.30), 0.5)}
N = {"n": 0}


def box(group, cx, cy, z0, sx, sy, sz, color, collide=True, name="b"):
    """바닥 높이 z0부터 크기 (sx, sy, sz) 상자. collide면 PhysX 충돌체 (지도 생성기·AGV 물리가 본다)."""
    N["n"] += 1
    path = f"/World/{group}/{name}_{N['n']}"
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.XformCommonAPI(c)
    xf.SetTranslate(Gf.Vec3d(cx, cy, z0 + sz / 2))
    xf.SetScale(Gf.Vec3f(sx, sy, sz))
    rgb, rough = C[color]
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat(color, rgb, rough, 0.6 if color in ("shelf", "upright", "beam") else 0.0))
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return path


# ── 건물: 바닥(충돌체), 벽, 지붕, 기둥(노란 방호대) ─────────────────────
box("marks", (X0 + X1) / 2, (Y0 + Y1) / 2, -0.2, X1 - X0 + 2, Y1 - Y0 + 2, 0.2, "floor", collide=False, name="floor")
gp = UsdGeom.Plane.Define(stage, "/World/fixed/ground_collision")   # 충돌은 무한 평면 (Simple_Warehouse의 GroundPlane과 같게 —
gp.CreateAxisAttr("Z")                                              # 상자 바닥은 바퀴가 ~1cm 파묻혀 끌리고 제자리 회전에서 멈췄다)
gp.CreatePurposeAttr("guide")
UsdPhysics.CollisionAPI.Apply(gp.GetPrim())
for cx, cy, sx, sy in (((X0 + X1) / 2, Y0 - 0.15, X1 - X0 + 0.6, 0.3), ((X0 + X1) / 2, Y1 + 0.15, X1 - X0 + 0.6, 0.3),
                       (X0 - 0.15, (Y0 + Y1) / 2, 0.3, Y1 - Y0), (X1 + 0.15, (Y0 + Y1) / 2, 0.3, Y1 - Y0)):
    box("fixed", cx, cy, 0.0, sx, sy, H, "wall", name="wall")
box("marks", (X0 + X1) / 2, (Y0 + Y1) / 2, H, X1 - X0 + 1, Y1 - Y0 + 1, 0.3, "roof", collide=False, name="roof")
for x, y in COLUMNS:
    box("fixed", x, y, 0.0, 0.6, 0.6, H, "column", name="column")
    box("fixed", x, y, 0.0, 0.8, 0.8, 1.0, "guard", name="column_guard")
for x in DOORS:                                                  # 도크 문(벽 안쪽 면) + 양옆 방호 기둥
    box("marks", x, Y0 + 0.02, 0.0, 3.0, 0.04, 3.6, "door", collide=False, name="dock_door")
    for dx in (-1.8, 1.8):
        box("fixed", x + dx, Y0 + 0.5, 0.0, 0.2, 0.2, 1.1, "guard", name="bollard")

# ── 동쪽: 팔레트 랙 (파란 기둥, 주황 빔) ──────────────────────────────
bays = 6
bay = (ROW_Y1 - ROW_Y0) / bays
levels = [0.0, 1.6, 3.1, 4.6]                                    # 바닥 + 빔 3단
for xa, xb in RACK_ROWS:
    for face in (xa, xb):                                        # 양면 — 바깥 면 기둥 줄과 안쪽(1.1m) 기둥 줄
        inner = face + 1.1 if face == xa else face - 1.1
        for k in range(bays + 1):
            y = ROW_Y0 + k * bay
            for x in (face, inner):
                box("fixed", x, y, 0.0, 0.09, 0.09, 6.0, "upright", name="rack_upright")
        for k in range(bays):
            yc = ROW_Y0 + (k + 0.5) * bay
            for z in levels[1:]:
                for x in (face, inner):
                    box("fixed", x, yc, z - 0.12, 0.05, bay - 0.09, 0.12, "beam", collide=z < 1.2, name="rack_beam")
            for z in levels:                                     # 칸마다 팔레트 2자리, 80% 채움
                for j in (-0.25, 0.25):
                    if rng.random() > 0.8:
                        continue
                    lx, ly = (face + inner) / 2, yc + j * bay
                    hh = rng.uniform(0.7, 1.15)
                    lo = z
                    box("props", lx, ly, lo, 1.0, 1.15, 0.14, "pallet", collide=lo < 1.2, name="rack_pallet")
                    box("props", lx, ly, lo + 0.14, 0.95, 1.1, hh, rng.choice(["box", "box2", "wrap"]), collide=lo < 1.2,
                        name="rack_load")

# ── 서쪽: 다단 선반 (회색, 5단) + 토트 ────────────────────────────────
units = 13
unit = (ROW_Y1 - ROW_Y0) / units
shelves = [0.12, 0.55, 1.0, 1.45, 1.9]
for xa, xb in SHELF_ROWS:
    xm = (xa + xb) / 2
    for k in range(units + 1):
        y = ROW_Y0 + k * unit
        for x in (xa + 0.02, xm, xb - 0.02):
            box("fixed", x, y, 0.0, 0.04, 0.04, 2.2, "shelf", name="shelf_post")
    for k in range(units):
        yc = ROW_Y0 + (k + 0.5) * unit
        for side in ((xa + xm) / 2, (xm + xb) / 2):
            for z in shelves:
                box("fixed", side, yc, z - 0.03, 0.58, unit - 0.04, 0.03, "shelf", collide=z < 1.2, name="shelf_plate")
                n = rng.choice([0, 1, 2, 2, 2])
                for j in range(n):
                    ty = yc + (j - (n - 1) / 2) * 0.55
                    box("props", side, ty, z, 0.55, 0.40, 0.30, rng.choice(["yellow_tote", "yellow_tote", "blue_tote", "gray_tote"]),
                        collide=z < 1.2, name="tote")

# ── 서쪽 벽: 컨베이어 + 포장대 ─────────────────────────────────────────
cx, cy0, cy1 = -32.6, -18.0, 6.0
box("fixed", cx, (cy0 + cy1) / 2, 0.0, 0.9, cy1 - cy0, 0.75, "conv", name="conveyor")
box("fixed", cx, (cy0 + cy1) / 2, 0.75, 0.75, cy1 - cy0, 0.06, "belt", collide=False, name="conveyor_belt")
y = cy0 + 0.6
while y < cy1 - 0.4:
    if rng.random() < 0.5:
        box("props", cx, y, 0.81, 0.55, 0.40, 0.30, rng.choice(["yellow_tote", "blue_tote"]), collide=False, name="conv_tote")
    y += 0.9
for ty in (-14.0, -10.0, -6.0, -2.0):
    box("fixed", -31.2, ty, 0.0, 0.9, 1.8, 0.9, "table", name="pack_table")
    box("fixed", -31.45, ty, 0.9, 0.08, 0.5, 0.35, "screen", collide=False, name="pack_screen")
    box("props", -31.0, ty + 0.5, 0.9, 0.4, 0.3, 0.25, "box2", collide=False, name="pack_box")

# ── 남쪽 입고장·빈 바닥의 물건 (지도에 없음 → 가서 본다) ───────────────
def clear(x, y, r):
    if not (X0 + 1 < x < X1 - 1 and Y0 + 1 < y < 7.0):
        return False
    if abs(x - ENTRANCE[0]) < 3.0 and y < -12.0:               # 입구 앞은 비움
        return False
    if any(abs(x - cx_) < r + 0.6 and abs(y - cy_) < r + 0.6 for cx_, cy_ in COLUMNS):
        return False
    return x > -29.5 and all((x - px) ** 2 + (y - py) ** 2 > (r + pr + 0.4) ** 2 for px, py, pr in placed)


placed = []
for _ in range(400):
    if len(placed) >= 10:
        break
    x, y = rng.uniform(X0 + 2, X1 - 2), rng.uniform(Y0 + 1.5, -13.0)   # 도크 앞 적치 팔레트
    if clear(x, y, 0.8):
        hh = rng.uniform(0.6, 1.5)
        box("props", x, y, 0.0, 1.2, 1.0, 0.14, "pallet", name="dock_pallet")
        box("props", x, y, 0.14, 1.15, 0.95, hh, rng.choice(["box", "box2", "wrap"]), name="dock_load")
        placed.append((x, y, 0.8))
for _ in range(300):
    if len(placed) >= 14:
        break
    x, y = rng.uniform(X0 + 3, X1 - 2), rng.uniform(-12.0, 6.0)     # 바닥에 놓인 상자·롤 케이지
    if clear(x, y, 0.5):
        if rng.random() < 0.5:
            box("props", x, y, 0.0, 0.8, 0.7, 1.7, "cage", name="roll_cage")
        else:
            for k in range(rng.randint(1, 3)):                          # 1~3개 쌓은 상자
                box("props", x + rng.uniform(-0.05, 0.05), y + rng.uniform(-0.05, 0.05), 0.35 * k, 0.5, 0.4, 0.35, "box", name="floor_box")
        placed.append((x, y, 0.5))
fk = stage.DefinePrim("/World/props/parked_forklift", "Xform")   # 세워 둔 지게차 (동쪽 빈 바닥)
fk.GetReferences().AddReference(ASSETS + "/Isaac/Props/Forklift/forklift.usd")
UsdGeom.XformCommonAPI(fk).SetTranslate(Gf.Vec3d(8.5, 12.0, 0.0))
UsdGeom.XformCommonAPI(fk).SetRotate(Gf.Vec3f(0, 0, 90))
box("fixed", 11.0, 20.0, 0.0, 1.2, 0.5, 1.2, "charger", name="agv_charger")
box("fixed", 11.0, 16.0, 0.0, 1.2, 0.5, 1.2, "charger", name="agv_charger")

# ── 바닥 표시 (충돌체 없음, 바닥 위 2mm) ────────────────────────────────
def line(x0, y0, x1, y1, w, color):
    if x0 == x1:
        box("marks", x0, (y0 + y1) / 2, 0.0, w, abs(y1 - y0), 0.004, color, collide=False, name="line")
    else:
        box("marks", (x0 + x1) / 2, y0, 0.0, abs(x1 - x0), w, 0.004, color, collide=False, name="line")


line(-30.0, 8.3, 4.0, 8.3, 0.12, "line_y")                       # 저장 구역 앞 통로 경계
line(-30.0, -12.5, 12.0, -12.5, 0.12, "line_y")                  # 입고장 경계
for zx, _ in ZONES.values():                                     # 통로 입구 정지선
    line(zx - 1.0, 8.8, zx + 1.0, 8.8, 0.3, "line_w")
line(-29.6, -18.0, -29.6, 6.0, 0.15, "line_g")                   # 포장대 앞 보행자 통로
line(-28.4, -18.0, -28.4, 6.0, 0.15, "line_g")
for x in (-25.0, -19.0, -13.0, -7.0, -1.0, 5.0, 11.0):           # 도크 앞 주차 칸
    line(x, -21.8, x, -18.5, 0.1, "line_y")

# ── 조명: 천장 LED 격자 + 약한 하늘빛 ──────────────────────────────────
k = 0
for x in range(-30, 14, 8):
    for y in range(-18, 27, 8):
        lt = UsdLux.RectLight.Define(stage, f"/World/lights/led_{k}")
        lt.CreateWidthAttr(2.4)
        lt.CreateHeightAttr(0.6)
        lt.CreateIntensityAttr(30000)
        UsdGeom.XformCommonAPI(lt).SetTranslate(Gf.Vec3d(x, y, H - 0.4))
        k += 1
UsdLux.DomeLight.Define(stage, "/World/lights/dome").CreateIntensityAttr(500)
stage.GetRootLayer().Save()

rows_x = sorted(SHELF_ROWS + RACK_ROWS)
aisles, slots = [], {}
for name, (zx, _) in ZONES.items():                                # 구역 통로 = 구역 x를 사이에 둔 두 줄 사이
    xl = max(xb for xa, xb in rows_x if xb < zx)
    xr = min(xa for xa, xb in rows_x if xa > zx)
    aisles.append({"name": name, "rect": [xl, ROW_Y0, xr, ROW_Y1], "width": round(xr - xl, 2),
                   "mouths": [[zx, ROW_Y0 - 0.5], [zx, ROW_Y1 + 0.5]], "axis": "y"})
    for k in range(int((ROW_Y1 - ROW_Y0 - 1.0) / 2.5)):
        slots[f"{name}-{k + 1:02d}"] = [zx, ROW_Y0 + 1.0 + 2.5 * k + 1.25]
qr = [[*p_, n] for n, p_ in slots.items()] + [[*m, f"{al['name']}-mouth{j}"] for al in aisles for j, m in enumerate(al["mouths"])]
qr += [[float(gx), float(gy), f"G{gx}_{gy}"] for gx in range(int(X0) + 2, int(X1) - 1, 4) for gy in range(int(Y0) + 2, 7, 4)]
env = {"usd": str(out / "fc_warehouse.usda"), "entrance": ENTRANCE, "entrance_yaw": 1.5707963, "zones": ZONES,
       "slots": slots, "aisles": aisles, "qr": qr,
       "routes": [["입구", z] for z in ZONES] + [["P-2", "R-2"], ["R-3", "P-1"]],
       "rack_rows": [[xa, ROW_Y0, xb, ROW_Y1] for xa, xb in SHELF_ROWS + RACK_ROWS],
       "props_root": "props", "bounds": [X0, Y0, X1, Y1]}
(out / "fc_env.json").write_text(json.dumps(env, ensure_ascii=False, indent=1))
print("prims", N["n"], "lights", k, "→", out / "fc_warehouse.usda")
