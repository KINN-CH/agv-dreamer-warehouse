#!/usr/bin/env python3
"""물류창고 AGV 주행 (5090 VM, Isaac Sim 6.1) — 내비게이션 + 돌발 상황 4종.

    python logistics.py --inspect --out ~/agv/logistics     # 창고 크기·물체 목록·위에서 본 그림·격자 지도
    python logistics.py --run --out ~/agv/logistics         # 돌발 없이 입구 → A-1~A-6, 구역 → 구역 (inspect의 격자 지도 사용)
    python logistics.py --hazards box,person,crossing,forklift --out ~/agv/logistics
        # 돌발 상황마다 같은 배치에서 대응 없음 vs 대응 — box 떨어진 상자 1~6개, person 서 있는 사람 1~3명,
        # crossing 가로지르는 사람, forklift 지나가는 지게차. 결과 <out>/hazards/<종류>/
    python logistics.py --collect 400 --first 0 --out ~/agv/logistics
        # 월드모델 학습 데이터 — 돌발 4종 무작위, 위험 지점 10m 앞 출발 16초, 행동 정책 섞음. <out>/wm_data/
"""
import argparse
import copy
import gzip
import json
import math
import os
import pickle
import re
import subprocess
import sys
import random
import time
from pathlib import Path

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="./logistics")
ap.add_argument("--inspect", action="store_true")
ap.add_argument("--racks-only", action="store_true", help="--inspect: 바닥의 물건(박스·팔레트·통·지게차 등)을 끄고 랙·벽만으로 지도 → grid_racks.npy")
ap.add_argument("--map", default="full", choices=["full", "racks"],
                help="AGV가 처음 아는 지도: full = 장면 전체, racks = 랙·벽만 (물건은 주행 중 카메라로 발견). 충돌 판정은 늘 전체")
ap.add_argument("--run", action="store_true")
ap.add_argument("--hazards", default="", help="쉼표로: box,person,crossing,forklift")
ap.add_argument("--conds", default="none,respond", help="--hazards 비교 조건, 쉼표로: none,respond,cv,wm")
ap.add_argument("--wm-ckpt", default="", help="cond wm: 월드모델 체크포인트 (src/wm_bev.py)")
ap.add_argument("--tag", default="", help="--hazards 결과 폴더 이름 뒤에 붙임")
ap.add_argument("--tau", type=float, default=0.1, help="앞보기: 2초 충돌 위험이 이보다 낮은 가장 빠른 속도를 고른다")
ap.add_argument("--vmax", type=float, default=1.0, help="경로 추종 최고 속도 m/s (속도만 낮춘 대조군용)")
ap.add_argument("--seeds", type=int, default=5, help="--hazards: 경로마다 배치 몇 가지")
ap.add_argument("--routes", type=int, default=6, help="--hazards: 입구 → A-1부터 몇 개 구역")
ap.add_argument("--collect", type=int, default=0, help="월드모델 학습 데이터 에피소드 수")
ap.add_argument("--first", type=int, default=0, help="--collect: 첫 에피소드 번호 (이어서 모을 때)")
ap.add_argument("--kinds", default="box,person,crossing,forklift", help="--collect: 돌발 종류 (쉼표로)")
ap.add_argument("--stream", action="store_true", help="맥북 Streaming Client로 실시간 보기 — 1배속, 화면 = AGV 뒤 카메라 (Isaac 하나만)")
ap.add_argument("--debug", action="store_true", help="주행 중 1초마다 속도·정지 이유 출력")
ap.add_argument("--v5", action="store_true", help="v5 세상: 깊이 카메라 3대(360°)·BEV 초록=모든 장애물 점·위치 추정 오차+바닥 QR 보정·물체 묶음 확장(사람 7·지게차 3·팔레트·롤 케이지)")
ap.add_argument("--collect5", type=int, default=0, help="v5 학습 데이터 에피소드 수 (--env 창고, 입구/칸 → 다른 통로의 칸, 최대 60초)")
ap.add_argument("--types", default="", help="--eval5에서 이 장면 종류만 (traffic,danger,blocked — 병렬로 나눌 때, --tag 따로)")
ap.add_argument("--decide", default="tau", choices=["tau", "score"], help="앞보기 속도 결정: tau = 위험 < --tau 중 가장 빠름, score = 2초 전진 − --lam × 위험이 가장 큰 속도 + 가속은 0.6초 연속 괜찮을 때만")
ap.add_argument("--lam", type=float, default=12.0, help="--decide score: 충돌 한 번 = 몇 m 손해")
ap.add_argument("--backout", action="store_true", help="좁은 곳(회전 반경 1.15m 안에 장애물)에서 제자리 회전 대신 똑바로 후진 (뒤 1m 확인)")
ap.add_argument("--hybrid", action="store_true", help="앞보기 + 규칙의 정지 구역(앞 1.5m)을 최후 안전장치로 (감속 구역은 앞보기가 대신)")
ap.add_argument("--pathcand", action="store_true", help="앞보기 후보 = 계획 경로를 그 속도로 2초 따라갈 때의 행동열 (곡률 유지 대신)")
ap.add_argument("--escape", action="store_true", help="--hybrid: 앞보기가 4초 넘게 정지만 고르는데 규칙 정지 구역이 비었으면 0.3m/s로 천천히 지나감 + 앞보기 후진은 뒤 확인")
ap.add_argument("--log-look", action="store_true", help="0.2초마다 상태(명령·정지 구역·길·뒤 막힘·물체 간격) + 앞보기 후보 위험·선택 + 1/30초 명령을 결과에 저장 (원인 분석용)")
ap.add_argument("--swept", action="store_true", help="회전 가능 판단을 차체가 쓸고 지나갈 자리로 (양방향) — 쓸데없는 후진 줄임")
ap.add_argument("--smooth", action="store_true", help="명령에 가속도 제한 (가속 0.8, 감속 2.0m/s², 회전 2.0rad/s²) — 덜컹거림 줄임")
ap.add_argument("--hyb-slow", default="", help="--hybrid: 확정 안 된 물체가 앞 D m(옆 1.2m) 안이면 V m/s까지만 (예 2.5:0.6) — 빠른 지게차 대비")
ap.add_argument("--reach", action="store_true", help="--collect5: 실제 세계(바닥 물건 포함)로 목적지까지 길이 없는 장면은 버림")
ap.add_argument("--data-dir", default="wm_data5b", help="--collect5: 저장 폴더 이름")
ap.add_argument("--real-motion", action="store_true", help="사람·지게차 움직임을 실제처럼: 둥글게 꺾기·가감속·커브 감속·서서 돌아서기, 최고 속도 사람 1.4·지게차 2.0m/s")
ap.add_argument("--eval5", action="store_true", help="v5 평가 (eval_scenes 장면 × --conds)")
ap.add_argument("--zones", default="", help="--hazards에서 이 구역들만 (쉼표로, 병렬로 나눠 돌릴 때 — --tag도 따로)")
ap.add_argument("--eval-seeds", default="0:3", help="--eval5 배치 번호 범위 a:b (기본 = 시험 세트 0~2, 영상 후보 찾기는 새 번호)")
ap.add_argument("--rule-first", action="store_true", help="--eval5: 규칙이 부딪힌 장면만 다른 조건도 돌린다 (영상 후보 찾기)")
ap.add_argument("--only", default="", help="--hazards에서 한 장면만: 구역:seed (예: A-3:1)")
ap.add_argument("--env", default="", help="다른 창고: build_fc.py가 만든 *_env.json (USD·입구·구역·경로·랙 줄). 비우면 Simple_Warehouse")
ap.add_argument("--show-eps", default="", help="수집한 에피소드 번호들(쉼표로)을 같은 조건으로 다시 돌려 발표용 4화면 영상으로")
ap.add_argument("--demo", action="store_true", help="직접 해 보기: --start·--goal·--types·--demo-seed로 장면 하나를 --conds마다 주행 (demo.sh)")
ap.add_argument("--start", default="입구", help="--demo 출발: '입구' 또는 칸 이름 (예 A1-01)")
ap.add_argument("--goal", default="", help="--demo 목적지 칸 이름 (예 A7-02)")
ap.add_argument("--demo-seed", type=int, default=1, help="--demo 무작위 배치 번호 (같은 번호 = 같은 장면)")
ap.add_argument("--kind", default="random", help="--demo 위험 상황에서 덮치는 종류 (random 또는 KINDS13 중 하나)")
ap.add_argument("--wait", type=int, default=0, help="--demo --stream: 출발 전 기다리는 초 (맥북에서 접속할 시간)")
ap.add_argument("--dump-vis", action="store_true", help="--demo: 0.2초마다 3인칭·앞 카메라·깊이·BEV·행동을 npz로 (발표 그림용)")
ap.add_argument("--top-cam", action="store_true", help="영상용 카메라를 AGV 위 9m 내려다보기로 (640×640, GIF는 이 화면만)")
ap.add_argument("--follow-cam", type=float, default=0.0, help="영상용 카메라가 AGV 위 H m를 따라가며 내려다봄 (창고 기준 방향 고정 = 길찾기 지도와 같은 방향, 천장보다 낮게)")
ap.add_argument("--gif-all", action="store_true", help="--eval5: 모든 장면·조건의 GIF 저장")
ap.add_argument("--fixed-cam", default="", help="영상용 카메라를 창고 바닥 고정 위치 X,Y,높이에서 내려다보기로 (예 -6.3,-10.8,10)")
args = ap.parse_args()

app = SimulationApp({"headless": True, "width": 1280, "height": 960, **({"hide_ui": False} if args.stream else {})})
if args.stream:                                                  # stream.sh와 같은 WebRTC (Tailscale 주소로 접속)
    from isaacsim.core.utils.extensions import enable_extension as _en
    if os.environ.get("AGV_STREAM_IP"):                             # 원격 보기 주소 (예: Tailscale IP) — 환경 변수로
        app.set_setting("/exts/omni.kit.livestream.app/primaryStream/publicIp", os.environ["AGV_STREAM_IP"])
    _en("omni.kit.livestream.app")
    app.update()

import nav                                                               # noqa: E402
from isaacsim.core.utils.extensions import enable_extension   # noqa: E402
for ext in ("isaacsim.core.api", "isaacsim.robot.wheeled_robots", "isaacsim.sensors.camera", "isaacsim.asset.gen.omap"):
    enable_extension(ext)
app.update()

import numpy as np                                                       # noqa: E402
import omni.physx                                                        # noqa: E402
import omni.usd                                                          # noqa: E402
from PIL import Image                                                    # noqa: E402
import omni.timeline                                                     # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdSkel       # noqa: E402
from isaacsim.asset.gen.omap.bindings import _omap                      # noqa: E402
from isaacsim.core.api import World                                      # noqa: E402
from isaacsim.core.prims import SingleXFormPrim                          # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat          # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage            # noqa: E402
from isaacsim.robot.wheeled_robots.controllers import DifferentialController  # noqa: E402
from isaacsim.robot.wheeled_robots.robots import WheeledRobot            # noqa: E402
from isaacsim.sensors.camera import Camera                              # noqa: E402
from isaacsim.storage.native import get_assets_root_path                # noqa: E402

FPS, CELL = 30, 0.05
ENV = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
Z_BAND = (0.05, 1.0)        # 격자 지도에 넣을 높이 — 바닥(0)은 빼고, AGV·카메라 높이대를 덮는다
# iw.hub 치수 (gen_episodes.py와 같음)
WHEEL_R, WHEEL_BASE = 0.081, 0.579
BODY_FRONT, BODY_BACK, BODY_HALF_W = 0.398, -1.033, 0.33
ROBOT_R = 0.7               # 계획 때 기준점이 장애물에서 떨어질 거리 — 차체 반폭 0.33 + 회전 때 뒤쪽이 안쪽으로 파고드는 몫
ENTRANCE = (-10.4, -18.0)   # 하역장 아래쪽 가운데, +y(랙 쪽)를 바라봄 (뒤따르는 카메라가 벽 안쪽에 오게 남쪽 벽에서 5.4m)
ZONES = {f"A-{k + 1}": (x, 17.0) for k, x in enumerate([-22.83, -17.93, -12.98, -7.98, -3.03, 1.95])}  # 랙 사이 통로 가운데
ROUTES = [("입구", z) for z in ZONES] + [("A-2", "A-5"), ("A-6", "A-1")]
ENV_JSON = json.loads(Path(args.env).expanduser().read_text()) if args.env else None
if ENV_JSON:                                                    # 다른 창고 (usd가 비면 Simple_Warehouse에 통로·칸·QR만 붙인 것)
    ENV, ENTRANCE = ENV_JSON["usd"] or ENV, tuple(ENV_JSON["entrance"])
    ZONES = {k: tuple(v) for k, v in ENV_JSON["zones"].items()}
    ROUTES = [tuple(r) for r in ENV_JSON["routes"]]


def env_url():
    return ENV if ENV_JSON and ENV_JSON["usd"] else get_assets_root_path() + ENV


AISLES = ENV_JSON.get("aisles") if ENV_JSON else None             # 통로 사각형·입구 — 돌발 배치를 창고 구조에 맞춘다
QR = np.array([q[:2] for q in ENV_JSON.get("qr", [])], float).reshape(-1, 2) if ENV_JSON else np.zeros((0, 2))
START_YAW = ENV_JSON.get("entrance_yaw", math.pi / 2) if ENV_JSON else math.pi / 2


def in_aisle(P, m=0.5):
    return any(r[0] - m <= P[0] <= r[2] + m and r[1] - m <= P[1] <= r[3] + m for r in (al["rect"] for al in AISLES or []))


def open_ok(P):
    """움직이는 돌발이 AGV와 만날 점 P — 통로 밖 빈 바닥 (Simple_Warehouse 예전 방식은 하역장 y −14~5)."""
    return not in_aisle(P, 0.8) if AISLES else -14.0 <= P[1] <= 5.0
TIMEOUT = 150.0                 # 실행당 최대 시뮬 시간(초)
SENSE_EVERY, REC_EVERY = 3, 6   # 3스텝마다 감지(10Hz), 6스텝마다 GIF 한 장(5fps → GIF 10fps로 2배속)
SENSE_Z, SENSE_RANGE = (0.1, 1.0), 5.0   # 장애물로 볼 점: 바닥 위 0.1~1.0m, 5m 안. 바닥 점(0.05m 아래)은 '비었음' 증거
PARK = np.array([0.0, 60.0, 0.0])        # 안 쓰는 물체를 치워 두는 곳(창고 밖)
# 떨어진 상자 — 창고 소품. 처음부터 바닥에 놓여 있다 (물리로 떨어뜨리지 않음, 움직이지 않는 충돌체)
BOX_TYPES = ["SM_CardBoxA_01", "SM_CardBoxB_01", "SM_CardBoxD_01"]
BOX_DIR = "/Isaac/Environments/Simple_Warehouse/Props/"
MAX_BOXES = 6
# 사람 — 동작 파일과 뼈대가 맞는 마네킹 (작업복 캐릭터는 뼈대 이름이 달라 동작이 안 붙는다)
PERSON_URL = "/Isaac/People/Characters/biped_demo/biped_demo_meters.usd"
ANIM = {"idle": "/Isaac/People/Animations/stand_idle_loop.skelanim.usd",
        "walk": "/Isaac/People/Animations/stand_walk_loop_in_place.skelanim.usd"}
WALK_SPEED = 0.54               # 걷기 동작의 이동 속도(m/s, stand_walk_loop의 Root 이동으로 잼) — 실제 속도 비율로 재생 빠르기를 맞춘다
PERSON_HALF, PERSON_YAW = (0.3, 0.25), math.pi / 2   # 사람 바닥 면적 반폭(진행 방향, 옆), 마네킹 앞이 자기 −y라서 +90°
MAX_PEOPLE = 3
POOL_PEOPLE, POOL_FORK = (7, 3) if args.v5 else (MAX_PEOPLE, 1)   # v5 사람: 덮치는 위험 0~2·양보 횡단 3·배경 4~6 / 지게차 0·1·2
CLUTTER = ["pallet_load", "cage"]                                 # v5 바닥 물건(지도에 없음): 짐 실린 팔레트, 롤 케이지
EXTRA_SENSE = []                                                 # v5: 뒤쪽 깊이 카메라 2대
FORK_URL = "/Isaac/Props/Forklift/forklift.usd"
V_AGV = 1.0                     # 움직이는 돌발의 출발 시점을 맞출 때 쓰는 AGV 속도 추정
SENSE_FOCAL = 0.6               # 감지용 깊이 카메라 초점거리 → 가로 화각 약 120° (녹화용 앞 카메라 60°와 따로)


def bbox(stage, path):
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
        stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


def occupancy(stage, lo, hi):
    """PhysX 충돌체로 2D 격자 지도. (격자[y, x] bool, 원점 x0, y0) — 칸 = CELL m."""
    gen = _omap.Generator(omni.physx.get_physx_interface(), omni.usd.get_context().get_stage_id())
    gen.update_settings(CELL, 1.0, 0.0, 0.5)
    gen.set_transform((0, 0, 0), (float(lo[0]), float(lo[1]), Z_BAND[0]), (float(hi[0]), float(hi[1]), Z_BAND[1]))
    gen.generate2d()
    pts = np.array(gen.get_occupied_positions(), dtype=np.float32).reshape(-1, 3)
    nx, ny = int(np.ceil((hi[0] - lo[0]) / CELL)), int(np.ceil((hi[1] - lo[1]) / CELL))
    grid = np.zeros((ny, nx), bool)
    ix = np.clip(((pts[:, 0] - lo[0]) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - lo[1]) / CELL).astype(int), 0, ny - 1)
    grid[iy, ix] = True
    return grid, float(lo[0]), float(lo[1])


PROPS = re.compile(r"^(Box|Bottle|Crate|SM_PaletteA|Barrel|S_TrafficCone|S_WetFloorSign|SM_PushcartA|SM_BucketPlastic|SM_CratePlastic"
                   r"|SM_CardBox|SM_BarelPlastic|SM_BottlePlastic|forklift|SM_PaperNote|SM_Paper_)")   # 창고 장면의 옮길 수 있는 물건


def inspect_racks(out: Path):
    """랙·벽·기둥만의 지도 — 장면의 물건(PROPS)을 꺼 두고 같은 범위·같은 칸으로 점유 지도. AGV가 '처음 아는 지도'."""
    world = World(stage_units_in_meters=1.0, physics_dt=1 / FPS, rendering_dt=1 / FPS)
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(env_url(), "/World/env")
    meta = json.loads((out / "inspect.json").read_text())
    props_root = ENV_JSON["props_root"] if ENV_JSON else None
    off = [p for p in stage.GetPrimAtPath("/World/env").GetChildren() if PROPS.match(p.GetName()) or p.GetName() == props_root]
    for p in off:
        p.SetActive(False)
    world.reset()                                                   # 물리 장면을 만들어야 지도 생성기가 충돌체를 본다
    for _ in range(10):
        world.step(render=False)
    grid, _, _ = occupancy(stage, np.array(meta["bbox_min"]), np.array(meta["bbox_max"]))
    assert grid.mean() > 0.001, f"지도가 비었다 ({grid.mean():.4f}) — 물리 장면 확인"
    np.save(out / "grid_racks_omap.npy", grid)                      # 물건을 끈 충돌체 — 벽·기둥·랙 끝 프레임뿐
    # 랙 바닥 면적은 통째로 막는다 (Simple_Warehouse 랙은 선반 충돌체가 없고, FC 랙은 빈 칸 사이로 길이 나므로)
    rows = [((x0, y0), (x1, y1)) for x0, y0, x1, y1 in ENV_JSON["rack_rows"]] if ENV_JSON and ENV_JSON.get("rack_rows") else \
        [(k["min"], k["max"]) for k in meta["children"] if re.match(r"^(SM_RackShelf|SM_RackPile|SM_Rackshield|SM_RackFrame)", k["name"])]
    for lo, hi in rows:
        lo, hi = np.array(lo), np.array(hi)
        j0, j1 = int((lo[0] - meta["x0"]) / CELL), int(np.ceil((hi[0] - meta["x0"]) / CELL))
        i0, i1 = int((lo[1] - meta["y0"]) / CELL), int(np.ceil((hi[1] - meta["y0"]) / CELL))
        grid[i0:i1, j0:j1] = True
    np.save(out / "grid_racks.npy", grid)
    Image.fromarray(np.where(grid[::-1], 0, 255).astype(np.uint8)).save(out / "grid_racks.png")
    print("racks-only: 끈 물건", len(off), "개 · grid", grid.shape, "occupied", round(grid.mean(), 4), flush=True)


def inspect(out: Path):
    world = World(stage_units_in_meters=1.0, physics_dt=1 / FPS, rendering_dt=1 / FPS)
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(env_url(), "/World/env")
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(800)
    lo, hi = bbox(stage, "/World/env")
    center = (lo + hi) / 2
    print("warehouse bbox min", lo.round(2), "max", hi.round(2), "size", (hi - lo).round(2), flush=True)
    kids = []
    for p in stage.GetPrimAtPath("/World/env").GetChildren():
        a, b = bbox(stage, p.GetPath())
        kids.append({"name": p.GetName(), "min": a.round(2).tolist(), "max": b.round(2).tolist()})
    cam = Camera(prim_path="/World/topcam", position=np.array([center[0], center[1], 45.0]), frequency=FPS,
                 resolution=(1280, 960), orientation=euler_angles_to_quat(np.array([0, 90, 0]), degrees=True))
    world.reset()
    cam.initialize()
    cam.set_focal_length(1.0)
    for _ in range(30):
        world.step(render=True)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cam.get_rgba()[:, :, :3].astype(np.uint8)).save(out / "topdown.png")
    grid, x0, y0 = occupancy(stage, lo, hi)
    np.save(out / "grid.npy", grid)
    Image.fromarray(np.where(grid[::-1], 0, 255).astype(np.uint8)).save(out / "grid.png")   # 위쪽 = +y
    meta = {"env": ENV, "cell": CELL, "x0": x0, "y0": y0, "z_band": Z_BAND, "bbox_min": lo.tolist(),
            "bbox_max": hi.tolist(), "occupied_frac": float(grid.mean()), "children": kids}
    (out / "inspect.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print("grid", grid.shape, "occupied", round(grid.mean(), 3), "children", len(kids), flush=True)


def body_clearance(m: nav.GridMap, x, y, yaw) -> float:
    """차체 직사각형 테두리에서 격자 지도 장애물까지 최단 거리. 0이면 닿은 것."""
    c, s = math.cos(yaw), math.sin(yaw)
    us, vs = np.arange(BODY_BACK, BODY_FRONT + 0.05, 0.05), np.arange(-BODY_HALF_W, BODY_HALF_W + 0.05, 0.05)
    pts = [(u, v) for u in us for v in (-BODY_HALF_W, BODY_HALF_W)] + [(u, v) for u in (BODY_BACK, BODY_FRONT) for v in vs]
    return min(m.clearance(x + c * u - s * v, y + s * u + c * v) for u, v in pts)


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


def pose(agv):
    p, (w, qx, qy, qz) = agv.get_world_pose()
    return float(p[0]), float(p[1]), math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def setup(hazards=False, chase_cam=True):
    """창고 + AGV + 앞 카메라 + 감지 카메라(깊이) + 뒤따르는 카메라(GIF용, 끌 수 있음) (+ 돌발 물체 묶음)."""
    world = World(stage_units_in_meters=1.0, physics_dt=1 / FPS, rendering_dt=1 / FPS)
    stage = omni.usd.get_context().get_stage()
    root = get_assets_root_path()
    add_reference_to_stage(env_url(), "/World/env")
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(800)
    add_reference_to_stage(root + "/Isaac/Robots/Idealworks/iwhub/iw_hub.usd", "/World/agv")
    agv = world.scene.add(WheeledRobot(prim_path="/World/agv", name="agv", create_robot=False,
                                       wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                                       position=np.array([*ENTRANCE, 0.02])))
    small = args.v5 and not chase_cam                               # v5 수집: 앞 RGB는 기록용 64×64면 된다
    front = Camera(prim_path="/World/agv/chassis/front_cam", translation=np.array([0.36, 0, 0.25]),
                   frequency=FPS, resolution=(64, 64) if small else (320, 240))
    chase = Camera(prim_path="/World/agv/chassis/chase_cam", translation=np.array([-4.5, 0, 2.6]), frequency=FPS,
                   resolution=(320, 240), orientation=euler_angles_to_quat(np.array([0, 22, 0]), degrees=True)) if chase_cam else None
    if chase_cam and args.fixed_cam:                                # 발표 영상용: 창고 바닥에 고정된 카메라로 내려다봄 (두 주행을 같은 화면에서 비교)
        fx_, fy_, fh_ = map(float, args.fixed_cam.split(","))
        chase = Camera(prim_path="/World/fixed_cam", position=np.array([fx_, fy_, fh_]), frequency=FPS,
                       resolution=(640, 640), orientation=euler_angles_to_quat(np.array([0, 90, 0]), degrees=True))
    elif chase_cam and args.follow_cam:                             # 발표용 긴 영상 (10-11): 출발부터 도착까지 AGV를 따라감
        chase = Camera(prim_path="/World/follow_cam", position=np.array([*ENTRANCE, args.follow_cam]), frequency=FPS,
                       resolution=(640, 640), orientation=euler_angles_to_quat(np.array([0, 90, 0]), degrees=True))
    elif chase_cam and args.top_cam:                                # 발표 영상용: AGV 위 9m에서 내려다봄 (충돌·간격이 분명하게)
        chase = Camera(prim_path="/World/agv/chassis/chase_cam", translation=np.array([0.3, 0, 9.0]), frequency=FPS,
                       resolution=(640, 640), orientation=euler_angles_to_quat(np.array([0, 90, 0]), degrees=True))
    SENSE_RES = (160, 120) if args.v5 else (320, 240)               # v5: 5m에서 점 간격 ~6.5cm — BEV 칸(15cm)엔 충분
    sense = Camera(prim_path="/World/agv/chassis/sense_cam", translation=np.array([0.36, 0, 0.25]),
                   frequency=FPS, resolution=SENSE_RES)
    pool = add_pool(stage, root) if hazards else {}
    EXTRA_SENSE.clear()
    if args.v5:                                                     # 왼쪽·오른쪽 옆면 가운데(±90°) + 뒷면(180°) — 앞 카메라와 합쳐 360°
        for name, tx, ty, yd in (("sense_l", -0.32, BODY_HALF_W + 0.01, 90.0), ("sense_r", -0.32, -BODY_HALF_W - 0.01, -90.0),
                                 ("sense_b", BODY_BACK - 0.01, 0.0, 180.0)):
            EXTRA_SENSE.append(Camera(prim_path=f"/World/agv/chassis/{name}", translation=np.array([tx, ty, 0.25]),
                                      frequency=FPS, resolution=SENSE_RES,
                                      orientation=euler_angles_to_quat(np.array([0, 0, yd]), degrees=True)))
    world.reset()
    for c in EXTRA_SENSE:
        c.initialize()
        c.set_focal_length(SENSE_FOCAL)
        c.add_distance_to_image_plane_to_frame()
        c.set_clipping_range(0.05, 100.0)                           # 기본(1m)이면 1m 안 물체가 안 보인다 (10-09 발견)
    for c in (front, chase, sense):
        if c is None:
            continue
        c.initialize()
        c.set_focal_length(1.8)
    sense.set_focal_length(SENSE_FOCAL)
    sense.add_distance_to_image_plane_to_frame()
    if args.v5:
        sense.set_clipping_range(0.05, 100.0)                       # 기본 clipping range (1m, 1e6)은 1m 안을 렌더하지 않는다
    ctrl = DifferentialController(name="ctrl", wheel_radius=WHEEL_R, wheel_base=WHEEL_BASE)
    if args.stream and chase is not None:                        # 스트림 화면 = AGV 뒤따르는 카메라
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().camera_path = "/World/agv/chassis/chase_cam"
    return world, agv, front, chase, sense, ctrl, pool


def _entry(stage, path, half_from_bbox=True, **kw):
    """창고 밖에 둔 물체 하나. half = 진행 방향 기준 반폭, off = 원점→바닥 면적 중심(물체 자기 좌표), zmin = 바닥 높이."""
    xf = SingleXFormPrim(path, name=path.split("/")[-1], position=PARK)
    lo, hi = bbox(stage, path)
    e = {"prim": xf, "off": ((hi[0] + lo[0]) / 2 - PARK[0], (hi[1] + lo[1]) / 2 - PARK[1]), "zmin": lo[2] - PARK[2],
         "yaw_off": 0.0, "anim": None}
    if half_from_bbox:
        e["half"] = ((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2)
    e.update(kw)
    return e


def add_pool(stage, root):
    """돌발 물체를 미리 만들어 창고 밖에 둔다. {종류: [물체, ...]}"""
    pool = {}
    for t in BOX_TYPES:                                             # 상자: 움직이지 않는 충돌체
        pool[t] = []
        for k in range(MAX_BOXES):
            path = f"/World/box_{t}_{k}"
            add_reference_to_stage(root + BOX_DIR + t + ".usd", path)
            prim = stage.GetPrimAtPath(path)
            for q in Usd.PrimRange(prim):
                if q.HasAPI(UsdPhysics.RigidBodyAPI):
                    q.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if not any(q.HasAPI(UsdPhysics.CollisionAPI) for q in Usd.PrimRange(prim)):
                for q in Usd.PrimRange(prim):
                    if q.IsA(UsdGeom.Mesh):
                        UsdPhysics.CollisionAPI.Apply(q)
                        UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr("convexHull")
            pool[t].append(_entry(stage, path))
        print(t, "바닥 면적(m)", np.round(2 * np.array(pool[t][0]["half"]), 2), flush=True)
    pool["person"] = []                                             # 사람: 동작을 붙인 마네킹 (충돌체 없음, 접촉은 기하로 판정)
    for k in range(POOL_PEOPLE):
        path = f"/World/person_{k}"
        add_reference_to_stage(root + PERSON_URL, path)
        skel = [q for q in Usd.PrimRange(stage.GetPrimAtPath(path)) if q.IsA(UsdSkel.Skeleton)][0]
        anim = stage.DefinePrim(f"/World/person_{k}_anim")
        anim.GetReferences().AddReference(root + ANIM["idle"])    # 묶기 전에 동작이 있어야 렌더에 붙는다
        UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
        pool["person"].append(_entry(stage, path, half_from_bbox=False, half=PERSON_HALF, off=(0.0, 0.0),
                                     yaw_off=PERSON_YAW, anim=anim))
    pool["forklift"] = []
    for k in range(POOL_FORK):
        path = "/World/forklift" if k == 0 else f"/World/forklift_{k}"
        add_reference_to_stage(root + FORK_URL, path)                # 지게차: 물리를 끄고 위치를 직접 옮긴다
        for q in Usd.PrimRange(stage.GetPrimAtPath(path)):
            for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.ArticulationRootAPI, UsdPhysics.CollisionAPI):
                if q.HasAPI(api):
                    q.RemoveAPI(api)
        e = _entry(stage, path)
        e["half"], e["yaw_off"] = e["half"][::-1], -math.pi / 2      # 긴 축이 자기 y → 진행 방향으로 돌린다
        pool["forklift"].append(e)
    if args.v5:                                                     # 바닥 물건: 상자 모양 충돌체 (움직이지 않음)
        for t, n_, parts in (("pallet_load", 6, [((1.2, 1.0, 0.14), 0.0, (0.62, 0.47, 0.30)), ((1.1, 0.95, 1.1), 0.14, (0.72, 0.56, 0.38))]),
                             ("cage", 3, [((0.8, 0.7, 1.7), 0.0, (0.62, 0.64, 0.66))])):
            pool[t] = []
            for k in range(n_):
                path = f"/World/clutter_{t}_{k}"
                UsdGeom.Xform.Define(stage, path)
                for j, ((sx, sy, sz), z0, rgb) in enumerate(parts):
                    cb = UsdGeom.Cube.Define(stage, f"{path}/p{j}")
                    cb.CreateSizeAttr(1.0)
                    UsdGeom.XformCommonAPI(cb).SetTranslate(Gf.Vec3d(0, 0, z0 + sz / 2))
                    UsdGeom.XformCommonAPI(cb).SetScale(Gf.Vec3f(sx, sy, sz))
                    cb.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
                    UsdPhysics.CollisionAPI.Apply(cb.GetPrim())
                pool[t].append(_entry(stage, path))
    print("forklift 바닥 면적(m)", np.round(2 * np.array(e["half"]), 2), flush=True)
    return pool


def corners(cx, cy, yaw, half):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + c * u - s * v, cy + s * u + c * v) for u, v in
            ((half[0], half[1]), (-half[0], half[1]), (-half[0], -half[1]), (half[0], -half[1]))]


def _perim(lo_u, hi_u, lo_v, hi_v, step=0.03):
    us = np.linspace(lo_u, hi_u, max(2, int((hi_u - lo_u) / step) + 1))
    vs = np.linspace(lo_v, hi_v, max(2, int((hi_v - lo_v) / step) + 1))
    return np.r_[np.c_[us, np.full_like(us, lo_v)], np.c_[us, np.full_like(us, hi_v)],
                 np.c_[np.full_like(vs, lo_u), vs], np.c_[np.full_like(vs, hi_u), vs]]


BODY_RECT = (BODY_BACK, BODY_FRONT, -BODY_HALF_W, BODY_HALF_W)
BODY_PERIM = _perim(*BODY_RECT)


def rect_gap(x, y, yaw, cx, cy, cyaw, half) -> float:
    """차체(x, y, yaw)와 물체 직사각형(중심, 방향, 반폭) 사이 최단 거리. 겹치면 0. 두 테두리를 서로의 좌표로 옮겨 잰다."""
    def world(pts, ox, oy, o):
        c, s = math.cos(o), math.sin(o)
        return np.c_[ox + c * pts[:, 0] - s * pts[:, 1], oy + s * pts[:, 0] + c * pts[:, 1]]

    def local(pts, ox, oy, o):
        c, s = math.cos(o), math.sin(o)
        d = pts - (ox, oy)
        return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]

    def dist(pts, lo_u, hi_u, lo_v, hi_v):
        du = np.maximum(np.maximum(lo_u - pts[:, 0], pts[:, 0] - hi_u), 0)
        dv = np.maximum(np.maximum(lo_v - pts[:, 1], pts[:, 1] - hi_v), 0)
        return float(np.hypot(du, dv).min())
    obj = (-half[0], half[0], -half[1], half[1])
    return min(dist(local(world(BODY_PERIM, x, y, yaw), cx, cy, cyaw), *obj),
               dist(local(world(_perim(*obj), cx, cy, cyaw), x, y, yaw), *BODY_RECT))


def sample_static(rng, path, fine, kinds, n, start, goal):
    """경로 위 1~2곳에 물체 n개를 흩뿌린다 — 경로 가운데서 옆으로 ±0.8m, 앞뒤로 ±1.2m, 방향 무작위.
    kinds = {종류: (반폭, 최대 개수)}. 랙·벽·서로 겹치지 않고 출발·도착 2.5m 안은 피한다. 반환 [(종류, x, y, 방향)]"""
    d = nav.densify(path, 0.1)
    if n == 0 or len(d) < 80:
        return []
    sites = [rng.randrange(35, len(d) - 35) for _ in range(1 if n <= 2 else rng.choice([1, 2]))]
    out, used = [], {t: 0 for t in kinds}
    for b in range(n):
        k = sites[b % len(sites)]
        tx, ty = d[min(k + 5, len(d) - 1)] - d[max(k - 5, 0)]
        tx, ty = tx / math.hypot(tx, ty), ty / math.hypot(tx, ty)
        for _ in range(100):
            t = rng.choice(sorted(kinds))
            rr = math.hypot(*kinds[t][0])
            off, along, yaw = rng.uniform(-0.8, 0.8), rng.uniform(-1.2, 1.2), rng.uniform(0, 2 * math.pi)
            cx, cy = d[k][0] + along * tx - off * ty, d[k][1] + along * ty + off * tx
            if (used[t] < kinds[t][1] and fine.clearance(cx, cy) >= rr + 0.05
                    and min(math.dist((cx, cy), start), math.dist((cx, cy), goal)) >= 2.5
                    and all(math.dist((cx, cy), (o[1], o[2])) >= rr + math.hypot(*kinds[o[0]][0]) for o in out)):
                out.append((t, cx, cy, yaw))
                used[t] += 1
                break
    return out


def sample_mover(rng, path, fine, half, lateral, speed, style="cross"):
    """움직이는 돌발 — 하역장(y −14~5) 경로 위 한 점 P에서 AGV와 만나게 움직인다.
    style cross: 경로 옆 lateral m에서 P를 직각으로 가로지른다 (서 있다가 출발).
    style turn: 경로 옆 lateral m 차선을 AGV 쪽으로 마주 오다가 P 옆에서 꺾어 가로지른다 (방향 바꾸는 사람·꺾는 지게차).
    AGV가 P에서 D m 안에 오면 출발하고, 대응이 없으면 P에서 거의 함께 만나게(±0.8m) D를 정한다.
    반환 way = 지나갈 점들 (a·b = 처음·끝)."""
    d = nav.densify(path, 0.1)
    for _ in range(500):
        if len(d) < 90:
            raise RuntimeError("경로가 짧음")
        k = rng.randrange(50, len(d) - 30)
        P = d[k]
        if not open_ok(P):
            continue
        tx, ty = d[min(k + 5, len(d) - 1)] - d[max(k - 5, 0)]
        tn = np.array([tx, ty]) / math.hypot(tx, ty)
        n = np.array([-tn[1], tn[0]])
        side, L, v = rng.choice([-1, 1]), rng.uniform(*lateral), rng.uniform(*speed)
        if style == "cross":
            way, to_P = [P - side * L * n, P + side * L * n], L
        else:
            ahead = rng.uniform(3.0, 6.0) * (1.6 if max(half) > 1.0 else 1.0)
            corner = P + side * L * n
            way, to_P = [corner + ahead * tn, corner, P - side * (L + 1.5) * n], ahead + L
        segs = list(zip(way, way[1:]))
        if min(fine.clearance(*(p0 + (p1 - p0) * u)) for p0, p1 in segs
               for u in np.linspace(0, 1, max(2, int(math.dist(p0, p1) / 0.1)))) < max(half) + 0.2:
            continue
        return {"way": [w.tolist() for w in way], "a": way[0].tolist(), "b": way[-1].tolist(), "speed": v, "P": P.tolist(),
                "D": max(1.0, to_P / v * V_AGV + rng.uniform(-0.8, 0.8))}
    raise RuntimeError("움직일 자리를 못 찾음")


def sample_mover2(rng, path, fine, half, style):
    """추가 시험 장면 — 멈추거나 비키거나 가려져 있는 움직임. 구간마다 속도 speeds, 그 구간을 시작하기 전 멈춤 waits(초).
    walkstop 옆에서 걸어와 경로 위에 서 버림 · hesitate 건너다 경로 위에서 멈칫 · follow 경로 위를 앞서 걷다 멈춤 ·
    oncoming 경로를 따라 마주 오다 직전에 옆으로 비킴 · forkstop 지게차가 가로지르다 경로를 막고 섬 ·
    emerge 옆 통로(랙 뒤)에 가려져 있다가 통로 입구로 나와 경로를 건넘.
    D = AGV가 P에서 D m 안에 오면 출발 — 대응 없이 1m/s로 가면 사건 때 AGV가 P 근처에 오게 정한다."""
    d = nav.densify(path, 0.1)
    for _ in range(800):
        if len(d) < 100:
            raise RuntimeError("경로가 짧음")
        k = rng.randrange(60, len(d) - 30)
        P = d[k]
        lo, hi = (5.5, 8.4) if style == "emerge" else (-14.0, 5.0)
        if (in_aisle(P, 0.3) if AISLES else not lo <= P[1] <= hi):
            continue
        tx, ty = d[min(k + 5, len(d) - 1)] - d[max(k - 5, 0)]
        tn = np.array([tx, ty]) / math.hypot(tx, ty)
        n = np.array([-tn[1], tn[0]])
        side = rng.choice([-1, 1])
        if style == "walkstop":                                   # 옆 3~4m에서 걸어와 경로 위(±0.3m)에 선다
            L, v = rng.uniform(3.0, 4.0), rng.uniform(0.9, 1.3)
            stop = P + rng.uniform(-0.3, 0.3) * n
            way, speeds, waits = [P - side * L * n, stop], [v], [0.0]
            t_event, lead = L / v, rng.uniform(1.0, 4.0)            # 서고 나서 AGV가 1~4m 남았을 때
        elif style == "hesitate":                                 # 건너다 경로 위에서 1.5~3초 멈칫, 다시 건넘
            L, v, pause = rng.uniform(3.0, 4.5), rng.uniform(1.0, 1.4), rng.uniform(1.5, 3.0)
            way, speeds, waits = [P - side * L * n, P, P + side * L * n], [v, v], [0.0, pause]
            t_event, lead = L / v, rng.uniform(-pause, 0.5)
        elif style == "follow":                                   # 경로 위 6~9m 앞에서 같은 방향으로 0.4~0.7m/s, 4~7m 걷고 섬
            v, ahead, walk = rng.uniform(0.4, 0.7), rng.uniform(6.0, 9.0), rng.uniform(4.0, 7.0)
            j = min(len(d) - 1, k + int(walk / 0.1))
            way, speeds, waits = [P, d[j]], [v], [0.0]
            t_event, lead = 0.0, ahead                              # AGV가 P에서 ahead m 뒤에 왔을 때 걷기 시작
        elif style == "oncoming":                                 # 경로 위 10~14m 앞에서 마주 오다, 2.5~4m 남기고 옆으로 1.2m 비켜 계속 감
            v, far, dodge = rng.uniform(0.9, 1.3), rng.uniform(10.0, 14.0), rng.uniform(2.5, 4.0)
            j = min(len(d) - 1, k + int(far / 0.1))
            mid = P + (d[j] - P) * 0.45
            way = [d[j], mid + side * 1.2 * n, P + side * 1.2 * n - tn * 3.0]
            speeds, waits = [v, v], [0.0, 0.0]
            t_event, lead = 0.0, far * 0.55 + dodge                 # 서로 다가오는 속도를 생각해 일찍 출발
        elif style == "forkstop":                                 # 지게차가 옆 7~9m에서 와서 경로 위에 멈춰 섬
            L, v = rng.uniform(7.0, 9.0), rng.uniform(1.5, 2.2)
            way, speeds, waits = [P - side * L * n, P], [v], [0.0]
            t_event, lead = L / v, rng.uniform(1.5, 4.0)
        else:                                                     # emerge: 옆 통로 안(랙 뒤)에서 내려와 통로 입구를 지나 경로를 건넘
            v = rng.uniform(1.0, 1.4)
            if AISLES:                                              # 창고 구조로: 경로에서 2.5~7.5m 떨어진 통로 입구 → 안쪽 2~3m에서 나와 건넘
                cands = [(al, m_) for al in AISLES for m_ in al["mouths"] if 2.5 <= math.dist(m_, P) <= 7.5]
                if not cands:
                    continue
                al, m_ = rng.choice(cands)
                r_ = al["rect"]
                inward = (np.array([0.0, math.copysign(1.0, (r_[1] + r_[3]) / 2 - m_[1])]) if al["axis"] == "y"
                          else np.array([math.copysign(1.0, (r_[0] + r_[2]) / 2 - m_[0]), 0.0]))
                corner = np.array(m_, float)
                start = corner + inward * rng.uniform(2.0, 3.0)
                way = [start, corner, P + (P - corner) / max(1e-6, float(np.linalg.norm(P - corner))) * 2.5]
                speeds, waits = [v, v], [0.0, 0.0]
                t_event, lead = (math.dist(start, corner) + math.dist(corner, P)) / v, rng.uniform(-0.6, 0.6)
                segs = list(zip(way, way[1:]))
                if min(fine.clearance(*(p0 + (p1 - p0) * u)) for p0, p1 in segs
                       for u in np.linspace(0, 1, max(2, int(math.dist(p0, p1) / 0.1)))) < 0.45:
                    continue
                return {"way": [np.asarray(w).tolist() for w in way], "a": np.asarray(way[0]).tolist(),
                        "b": np.asarray(way[-1]).tolist(), "speeds": speeds, "waits": waits, "speed": float(np.mean(speeds)),
                        "P": P.tolist(), "D": max(1.0, t_event * V_AGV + lead)}
            near = [x for x, _ in ZONES.values() if 2.5 <= abs(x - P[0]) <= 7.5]   # 옆 통로 가운데 (구역 x좌표)
            if not near:
                continue
            xa = rng.choice(near)
            side = 1 if xa > P[0] else -1
            start = np.array([xa, rng.uniform(10.0, 12.0)])
            corner = np.array([xa, P[1]])
            gap_x = xa - P[0]
            way = [start, corner, np.array([P[0] - side * 2.5, P[1]])]
            speeds, waits = [v, v], [0.0, 0.0]
            t_event, lead = (math.dist(start, corner) + abs(gap_x)) / v, rng.uniform(-0.6, 0.6)
        segs = list(zip(way, way[1:]))
        need = max(half) + 0.2 if style != "emerge" else 0.45
        if min(fine.clearance(*(p0 + (p1 - p0) * u)) for p0, p1 in segs
               for u in np.linspace(0, 1, max(2, int(math.dist(p0, p1) / 0.1)))) < need:
            continue
        return {"way": [np.asarray(w).tolist() for w in way], "a": np.asarray(way[0]).tolist(), "b": np.asarray(way[-1]).tolist(),
                "speeds": speeds, "waits": waits, "speed": float(np.mean(speeds)), "P": P.tolist(),
                "D": max(1.0, t_event * V_AGV + lead)}
    raise RuntimeError("움직일 자리를 못 찾음: " + style)


MOVERS = {   # 종류: (물체, 움직임, 경로에서 옆 거리 m, 속도 m/s)
    "crossing": ("person", "cross", (3.0, 5.0), (1.0, 1.6)),
    "forklift": ("forklift", "cross", (7.0, 10.0), (1.5, 2.5)),
    "stopgo": ("person", "cross", (1.2, 2.0), (1.0, 1.5)),          # 경로 바로 옆에 서 있다가 갑자기 건넘
    "turn": ("person", "turn", (1.5, 2.5), (1.0, 1.5)),             # 옆을 마주 걸어오다 꺾어 들어옴
    "forklift_turn": ("forklift", "turn", (2.5, 3.5), (1.5, 2.5)),  # 옆 차선을 마주 오다 꺾어 가로지름
}
MOVERS2 = {"walkstop": "person", "hesitate": "person", "follow": "person", "oncoming": "person",
           "forkstop": "forklift", "emerge": "person"}                 # 추가 시험 장면 (sample_mover2)


def along(way, s):
    """꺾은선 way를 따라 s m 간 점, 그 구간 방향, 가는 중인지."""
    for p0, p1 in zip(way, way[1:]):
        L = math.dist(p0, p1)
        if s <= L:
            u = max(s, 0.0) / L
            return (p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u), math.atan2(p1[1] - p0[1], p1[0] - p0[0]), s > 0
        s -= L
    p0, p1 = way[-2], way[-1]
    return (p1[0], p1[1]), math.atan2(p1[1] - p0[1], p1[0] - p0[0]), False


MOTION = {"person": dict(a=1.0, d=1.5, R=0.5, lat=1.0, w=2.0, vmax=1.4, body=0.3),     # 사람: 가속·감속 m/s², 꺾는 반경 m,
          "forklift": dict(a=0.6, d=1.0, R=1.5, lat=0.8, w=0.8, vmax=2.0, body=0.65)}   # 옆 가속도 m/s², 돌아서기 rad/s, 최고 m/s


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def realize(m, kind, fine):
    """꺾은선 움직임 m을 실제처럼 — 꼭짓점은 둥근 호(멈춤·후진 전환 꼭짓점은 서서 돌아섬), 가감속·커브 감속, 최고 속도.
    결과는 m["tr"] = 시간별 (t, x, y, 방향, 속도) 배열 — along_t가 이것을 따른다. P·D(만남 타이밍)가 있으면 가속 때문에 늦어진 만큼 D를 늘린다."""
    lim = MOTION[kind]
    way = [np.asarray(q, float) for q in (m["way"] if "way" in m else [m["a"], m["b"]])]
    n = len(way) - 1
    sp = [min(v, lim["vmax"]) for v in (m["speeds"] if "speeds" in m else [m["speed"]] * n)]
    wt = list(m["waits"]) if "waits" in m else [0.0] * n
    rev = list(m.get("rev") or [False] * n)
    d_ = [way[i + 1] - way[i] for i in range(n)]
    L = [float(np.hypot(*q)) for q in d_]
    u = [q / max(l, 1e-6) for q, l in zip(d_, L)]
    trim = [0.0] * (n + 1)                                          # 꼭짓점마다 둥글게 잘라낼 길이
    rad = [0.0] * (n + 1)
    sharp = [False] * (n + 1)                                       # 서서 돌아서는 꼭짓점
    for j in range(1, n):
        th = abs(_wrap(math.atan2(u[j][1], u[j][0]) - math.atan2(u[j - 1][1], u[j - 1][0])))
        if wt[j] > 0 or rev[j] != rev[j - 1] or th > math.radians(120):
            sharp[j] = True                                         # 멈추는 곳·후진 전환·120° 넘는 꺾기는 서서 돌아섬
            continue
        if th < math.radians(5):
            continue                                                # 거의 직선 — 그냥 지나감
        for R in (lim["R"], lim["R"] / 2, lim["R"] / 4):
            dd = min(R * math.tan(th / 2), 0.45 * L[j - 1], 0.45 * L[j])
            Re = dd / math.tan(th / 2)
            if Re < lim["R"] / 4 - 1e-6:                            # 구간이 짧아 호가 너무 작음 (가면서 제자리처럼 돎) → 서서 돌아섬
                sharp[j] = True
                break
            a0 = way[j] - u[j - 1] * dd                             # 호 시작
            nrm = np.array([-u[j - 1][1], u[j - 1][0]]) * math.copysign(1, np.cross(u[j - 1], u[j]))
            c = a0 + nrm * Re
            mid = c + (way[j] - c) / max(np.linalg.norm(way[j] - c), 1e-6) * Re   # 호 가운데 — 몸이 장애물에 안 닿게
            if fine.clearance(*mid) >= lim["body"] or R == lim["R"] / 4:
                trim[j], rad[j] = dd, Re
                break
    pts, seg, curv, stops = [], [], [], {}                          # 0.05m마다 점, 그 점의 구간·곡률, 멈춤(대기 초)
    ds = 0.05
    for i in range(n):
        s0, s1 = trim[i], L[i] - trim[i + 1]
        k_ = max(1, int((s1 - s0) / ds))
        for k in range(k_ + 1):
            if k == 0 and pts and i > 0 and trim[i] == 0:
                continue
            pts.append(way[i] + u[i] * (s0 + (s1 - s0) * k / k_))
            seg.append(i)
            curv.append(0.0)
        if i + 1 < n and trim[i + 1] > 0:                           # 다음 꼭짓점의 호
            a0, a1 = way[i + 1] - u[i] * trim[i + 1], way[i + 1] + u[i + 1] * trim[i + 1]
            sgn = math.copysign(1, np.cross(u[i], u[i + 1]))
            c = a0 + np.array([-u[i][1], u[i][0]]) * sgn * rad[i + 1]
            ang0, ang1 = math.atan2(*(a0 - c)[::-1]), math.atan2(*(a1 - c)[::-1])
            dang = _wrap(ang1 - ang0)
            k_ = max(2, int(abs(dang) * rad[i + 1] / ds))
            for k in range(1, k_):
                ang = ang0 + dang * k / k_
                pts.append(c + rad[i + 1] * np.array([math.cos(ang), math.sin(ang)]))
                seg.append(i + 1)
                curv.append(1 / rad[i + 1])
        elif i + 1 < n and sharp[i + 1]:
            stops[len(pts) - 1] = wt[i + 1]                         # 날카로운 꼭짓점: 서서 (대기 + 돌아서기)
    P_ = np.array(pts)
    if len(P_) < 2:
        return m
    steps = np.hypot(*np.diff(P_, axis=0).T)
    vt = np.array([min(sp[s], math.sqrt(lim["lat"] / c) if c > 0 else 9.0) for s, c in zip(seg, curv)])
    vt[0] = vt[-1] = 0.0
    for k in stops:
        vt[k] = 0.0
    v = vt.copy()
    for k in range(1, len(v)):                                      # 가속 제한 (앞으로)
        v[k] = min(v[k], math.sqrt(v[k - 1] ** 2 + 2 * lim["a"] * steps[k - 1]))
    for k in range(len(v) - 2, -1, -1):                             # 감속 제한 (뒤로)
        v[k] = min(v[k], math.sqrt(v[k + 1] ** 2 + 2 * lim["d"] * steps[k]))
    hd = [math.atan2(*((P_[k] - P_[k - 1]) if k in stops else (P_[min(k + 1, len(P_) - 1)] - P_[max(k - 1, 0)]))[::-1])   # 서는 꼭짓점은 오던 방향
          + (math.pi if rev[seg[k]] else 0.0) for k in range(len(P_))]                                # (가운데 차분은 U턴에서 0벡터 → 서기 전에 휙 돎)
    T, X, Y, Hh, V = [0.0], [P_[0][0]], [P_[0][1]], [hd[0]], [0.0]
    if wt[0] > 0:
        T.append(wt[0]); X.append(P_[0][0]); Y.append(P_[0][1]); Hh.append(hd[0]); V.append(0.0)
    for k in range(1, len(P_)):
        if k - 1 in stops or (k > 1 and seg[k] != seg[k - 1] and rev[seg[k]] != rev[seg[k - 1]]):   # 서서 대기·돌아서기
            dth = abs(_wrap(hd[k] - Hh[-1]))
            hold = max(stops.get(k - 1, 0.0), dth / lim["w"])
            T.append(T[-1] + hold); X.append(X[-1]); Y.append(Y[-1]); Hh.append(hd[k] if dth > 0 else Hh[-1]); V.append(0.0)
        T.append(T[-1] + 2 * steps[k - 1] / max(v[k - 1] + v[k], 1e-3)); X.append(P_[k][0]); Y.append(P_[k][1]); Hh.append(hd[k]); V.append(v[k])
    m["tr"] = {"t": T, "x": X, "y": Y, "h": Hh, "v": V}
    if "P" in m and m.get("D", 1e9) < 1e8:                          # 만남 타이밍: P 가장 가까이 오는 시각이 늦어진 만큼 D를 늘림
        P0 = np.asarray(m["P"], float)
        kP = int(np.argmin(np.hypot(np.array(X) - P0[0], np.array(Y) - P0[1])))
        t_prof = T[kP]
        s_cum, t_con, done_ = 0.0, wt[0], False
        for i in range(n):
            q = way[i] + u[i] * np.clip(np.dot(P0 - way[i], u[i]), 0, L[i])
            if np.hypot(*(q - P0)) < 0.3 and not done_:
                t_con += np.dot(q - way[i], u[i]) / max(sp[i], 1e-3)
                done_ = True
                break
            t_con += L[i] / max(sp[i], 1e-3) + (wt[i + 1] if i + 1 < n else 0.0)
        if done_:
            m["D"] = max(1.0, m["D"] + V_AGV * (t_prof - t_con))
    return m


def realize_all(acts, pool, fine):
    """장면의 움직이는 사람·지게차를 모두 실제 같은 움직임으로 (--real-motion)."""
    for a in acts:
        if a.get("move") and "tr" not in a["move"]:              # 두 번 부르면 D를 두 번 늘리므로 한 번만
            realize(a["move"], "forklift" if a["kind"] == "forklift" else "person", fine)
            (a["x"], a["y"]), a["heading"], _ = along_t(a["move"], 0.0)
    return acts


def along_t(m, tau):
    """움직임 m을 출발 후 tau초에: 위치, 방향, 걷는 중인지. 구간마다 멈춤(waits)·속도(speeds)가 있으면 따른다.
    m["tr"](실제 같은 움직임)이 있으면 그 시간표를 보간한다."""
    tr = m.get("tr")
    if tr is not None:
        T = tr["t"]
        k = int(np.searchsorted(T, tau, side="right")) - 1
        if k < 0:
            m["_v"] = 0.0
            return (tr["x"][0], tr["y"][0]), tr["h"][0], False
        if k >= len(T) - 1:
            m["_v"] = 0.0
            return (tr["x"][-1], tr["y"][-1]), tr["h"][-1], False
        f = (tau - T[k]) / max(T[k + 1] - T[k], 1e-6)
        h0, h1 = tr["h"][k], tr["h"][k + 1]
        vv = tr["v"][k] + (tr["v"][k + 1] - tr["v"][k]) * f
        m["_v"] = vv
        return ((tr["x"][k] + (tr["x"][k + 1] - tr["x"][k]) * f, tr["y"][k] + (tr["y"][k + 1] - tr["y"][k]) * f),
                h0 + _wrap(h1 - h0) * f, vv > 0.05)
    way = m["way"] if "way" in m else [m["a"], m["b"]]
    if "speeds" not in m:
        return along(way, m["speed"] * tau)
    rev, face_prev, hd_prev = m.get("rev"), m.get("face_prev"), None   # rev: 구간마다 후진(앞은 반대), face_prev: 멈춤 때 오던 방향 유지
    for i, ((p0, p1), v, wt) in enumerate(zip(zip(way, way[1:]), m["speeds"], m["waits"])):
        hd = math.atan2(p1[1] - p0[1], p1[0] - p0[0]) + (math.pi if rev and rev[i] else 0.0)
        if tau < wt:
            return (p0[0], p0[1]), (hd_prev if face_prev and hd_prev is not None else hd), False   # 멈춰 있음
        tau -= wt
        L = math.dist(p0, p1)
        if v * tau < L:
            u = max(v * tau, 0.0) / L
            return (p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u), hd, tau > 0
        tau -= L / v
        hd_prev = hd
    p0, p1 = way[-2], way[-1]
    return (p1[0], p1[1]), (hd_prev if hd_prev is not None else math.atan2(p1[1] - p0[1], p1[0] - p0[0])), False


def natural(rng, path, fine, pool, start, goal, n_walk=2, n_pick=1, n_fork=1, p0=0, f0=0, yg=(0.5, 0.5), boxes=True):
    """자연스러운 창고 (시험 전용 — 학습 데이터에 안 넣음). AGV와 상관없이 처음부터 각자 일한다:
    걷는 작업자 2명(경로 1~5m 안 통로를 A*로 오가며 70%는 2~8초 멈춤), 랙 보는 작업자 1명(랙을 보고 서 있다가 가끔 0.6~1.2m 물러났다 돌아옴),
    지게차 1대(랙 앞에 서서 3~5초 넣고 빼기 → 2.5m 후진 → 다음 랙), 바닥 상자 0~2개.
    사람·지게차는 1초 뒤 자리가 AGV에 0.5m 안이면 기다린다(양보, drive) — 충돌은 AGV가 들이받은 것만 남는다."""
    d = nav.densify(path, 0.1)
    cm = fine.coarse(0.1)
    far = {"P": list(start), "D": 1e9, "yield": True, "face_prev": True}   # D가 크면 처음부터 움직인다

    def spot(lo, hi, c_lo, c_hi=99.0):
        for _ in range(3000):
            p = d[rng.randrange(min(40, len(d) - 1), len(d))]       # 출발 4m 안은 빼고 경로 둘레
            a, r = rng.uniform(0, 2 * math.pi), rng.uniform(lo, hi)
            q = (float(p[0] + r * math.cos(a)), float(p[1] + r * math.sin(a)))
            if c_lo <= fine.clearance(*q) <= c_hi and math.dist(q, start) > 3.0:
                return q
        raise RuntimeError("자연 장면: 자리를 못 찾음")

    def facing(q):                                                  # 가장 가까운 랙 쪽 방향
        angs = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        return float(angs[int(np.argmin([fine.clearance(q[0] + 1.0 * math.cos(u), q[1] + 1.0 * math.sin(u)) for u in angs]))])

    def leg(a, b, r):                                               # A* 꺾은선 (시작점 빼고)
        p = nav.plan(cm, a, b, robot_r=r)
        return [tuple(map(float, q)) for q in nav.shortcut(cm, p, r)[1:]] if p else []

    acts = []
    for i in range(n_walk):                                         # 걷는 작업자
        way, speeds, waits, T = [spot(1.0, 5.0, 0.6)], [], [], 0.0
        for _ in range(60):
            if T > TIMEOUT:
                break
            seg = leg(way[-1], spot(1.0, 5.0, 0.6), 0.45)
            v, w = rng.uniform(0.9, 1.4), (rng.uniform(2.0, 8.0) if rng.random() < 0.7 else 0.0)
            for k, q in enumerate(seg):
                T += math.dist(way[-1], q) / v + (w if k == 0 else 0.0)
                way.append(q)
                speeds.append(v)
                waits.append(w if k == 0 else 0.0)
        if len(way) < 2:
            raise RuntimeError("자연 장면: 작업자 길 없음")
        acts.append({"kind": "person", "i": p0 + i, "move": {"way": way, "speeds": speeds, "waits": waits, "speed": 1.1, **far,
                                                             "yield_gap": rng.uniform(*yg)}})
    for i in range(n_pick):                                         # 랙 보는 작업자
        acts.append(_picker(rng, fine, spot, facing, p0 + n_walk + i, far, yg))
    if n_fork:
        acts.append(_forklift_worker(rng, fine, pool, spot, facing, leg, f0, far, yg))
    for a in acts:
        (a["x"], a["y"]), a["heading"], _ = along_t(a["move"], 0.0)
    if boxes:
        kinds = {t: (pool[t][0]["half"], MAX_BOXES) for t in BOX_TYPES}
        k = {}
        for t, x, y, yaw in sample_static(rng, path, fine, kinds, rng.randint(0, 2), start, goal):
            acts.append({"kind": t, "i": k.get(t, 0), "x": x, "y": y, "heading": yaw})
            k[t] = k.get(t, 0) + 1
    return acts


def _picker(rng, fine, spot, facing, i, far, yg):
    """랙을 보고 서 있다가 가끔 0.6~1.2m 물러났다 돌아오는 작업자."""
    q = spot(1.0, 3.5, 0.45, 0.8)
    f = facing(q)
    fx, fy = math.cos(f), math.sin(f)
    way, speeds, waits = [(q[0] - 0.3 * fx, q[1] - 0.3 * fy), q], [0.6], [0.0]
    while sum(waits) < TIMEOUT:
        L = rng.uniform(0.6, 1.2)
        back = (q[0] - L * fx, q[1] - L * fy)
        if fine.clearance(*back) < 0.4:
            break
        way += [back, q]
        speeds += [0.8, 0.8]
        waits += [rng.uniform(5.0, 15.0), rng.uniform(2.0, 5.0)]
    return {"kind": "person", "i": i, "move": {"way": way, "speeds": speeds, "waits": waits, "speed": 0.8, **far,
                                               "yield_gap": rng.uniform(*yg)}}


def _forklift_worker(rng, fine, pool, spot, facing, leg, i, far, yg):
    """랙 앞에 서서 넣고 빼기(3~5초) → 2.5m 후진 → 다음 랙으로 가는 지게차."""
    fh = pool["forklift"][0]["half"]                                # 지게차: (1.75, 0.6)
    rf = fh[1] + 0.35
    way = [spot(2.0, 6.0, rf + 0.2)]
    speeds, waits, rev = [], [], []
    for _ in range(8):
        if sum(math.dist(a, b) for a, b in zip(way, way[1:])) / 1.5 + sum(waits) > TIMEOUT:
            break
        face = spot(1.0, 6.0, fh[0] + 0.2, fh[0] + 0.6)            # 앞(포크)이 랙에 닿을 만한 자리
        f = facing(face)
        back = (face[0] - 2.5 * math.cos(f), face[1] - 2.5 * math.sin(f))
        if fine.clearance(*back) < rf:
            continue
        seg = leg(way[-1], back, rf)
        if not seg:
            continue
        v = rng.uniform(1.2, 2.0)
        for k, p in enumerate(seg):                                 # 다음 랙 앞까지 전진
            way.append(p)
            speeds.append(v)
            waits.append(rng.uniform(0.5, 2.0) if k == 0 and len(way) > 2 else 0.0)
            rev.append(False)
        way += [face, back]                                         # 랙으로 들어가 3~5초(넣기·꺼내기) → 후진으로 빠짐
        speeds += [0.6, 0.6]
        waits += [0.5, rng.uniform(3.0, 5.0)]
        rev += [False, True]
    if len(way) < 2:
        raise RuntimeError("자연 장면: 지게차 길 없음")
    return {"kind": "forklift", "i": i, "move": {"way": way, "speeds": speeds, "waits": waits, "rev": rev, "speed": 1.5, **far,
                                                 "yield_gap": rng.uniform(*yg)}}


def scenario(kind, rng, path, fine, pool, start, goal):
    """돌발 배치. 반환 [{kind, entry 번호, x, y, heading, (move)}]"""
    if kind == "natural":
        return natural(rng, path, fine, pool, start, goal)
    if kind == "box":
        kinds = {t: (pool[t][0]["half"], MAX_BOXES) for t in BOX_TYPES}
        objs = sample_static(rng, path, fine, kinds, rng.randint(1, MAX_BOXES), start, goal)
    elif kind == "person":
        objs = sample_static(rng, path, fine, {"person": (PERSON_HALF, MAX_PEOPLE)}, rng.randint(1, MAX_PEOPLE), start, goal)
    elif kind in MOVERS2:
        obj = MOVERS2[kind]
        m = sample_mover2(rng, path, fine, PERSON_HALF if obj == "person" else pool["forklift"][0]["half"], kind)
        (x, y), hd, _ = along_t(m, 0.0)
        return [{"kind": obj, "i": 0, "x": x, "y": y, "heading": hd, "move": m}]
    else:
        obj, style, lat, sp = MOVERS[kind]
        m = sample_mover(rng, path, fine, PERSON_HALF if obj == "person" else pool["forklift"][0]["half"], lat, sp, style)
        (x, y), hd, _ = along(m["way"], 0.0)
        return [{"kind": obj, "i": 0, "x": x, "y": y, "heading": hd, "move": m}]
    k = {}
    acts = []
    for t, x, y, yaw in objs:
        acts.append({"kind": t, "i": k.get(t, 0), "x": x, "y": y, "heading": yaw})
        k[t] = k.get(t, 0) + 1
    return acts


def place(e, x, y, heading):
    """바닥 면적 중심이 (x, y), 진행 방향이 heading이 되게 놓는다."""
    yaw = heading + e["yaw_off"]
    c, s = math.cos(yaw), math.sin(yaw)
    ox, oy = e["off"]
    e["prim"].set_world_pose(position=np.array([x - (c * ox - s * oy), y - (s * ox + c * oy), -e["zmin"] + 0.002]),
                             orientation=yaw_quat(yaw))


def set_anim(e, name, tau, stage_tc):
    """사람 동작 name을 동작 시각 tau(프레임)로 — 참조의 시간 어긋남(layer offset)으로 맞춘다 (타임라인과 무관)."""
    root = get_assets_root_path()
    e["anim"].GetReferences().SetReferences([Sdf.Reference(root + ANIM[name], layerOffset=Sdf.LayerOffset(offset=stage_tc - tau))])


def anim_len(e, name) -> float:
    if name not in e.setdefault("len", {}):
        set_anim(e, name, 0.0, 0.0)
        ts = UsdSkel.Animation(e["anim"]).GetRotationsAttr().GetTimeSamples()
        e["len"][name] = ts[-1] - ts[0] if ts else 1.0
    return e["len"][name]


def arrange(pool, acts):
    """이번 장면에 쓸 물체를 놓고 나머지는 치운다."""
    used = {(a["kind"], a["i"]) for a in acts}
    for t, es in pool.items():
        for i, e in enumerate(es):
            if (t, i) not in used:
                e["prim"].set_world_pose(position=PARK, orientation=yaw_quat(0))
    for a in acts:
        place(pool[a["kind"]][a["i"]], a["x"], a["y"], a["heading"])


def sense_points(cam, x, y, yaw, est=None):
    """감지 카메라 깊이 → 월드 점. (장애물 x, y: 바닥 위 0.1~1.0m·5m 안·차체 밖, 바닥 x, y: 5m 안 바닥 점 일부).
    est = 추정 자세 (ex, ey, eyaw)면 점을 추정 좌표로 옮긴다 — 로봇 기준 위치는 정확하고 지도와의 겹침만 어긋난다."""
    p = np.asarray(cam.get_pointcloud(world_frame=True))
    if p.ndim != 2 or not len(p):
        return np.zeros((0, 2)), np.zeros((0, 2))
    p = p[np.isfinite(p).all(1)]
    near = np.hypot(p[:, 0] - x, p[:, 1] - y) < SENSE_RANGE
    o = p[near & (p[:, 2] > SENSE_Z[0]) & (p[:, 2] < SENSE_Z[1])]
    dx, dy = o[:, 0] - x, o[:, 1] - y
    u, v = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    o = o[~((u > BODY_BACK - 0.1) & (u < BODY_FRONT + 0.1) & (np.abs(v) < BODY_HALF_W + 0.1))]
    f = p[near & (p[:, 2] < 0.05)][::2 if args.v5 else 8]
    o, f = o[:, :2], f[:, :2]
    if est is not None and tuple(est) != (x, y, yaw):
        def to_est(q):
            dx_, dy_ = q[:, 0] - x, q[:, 1] - y
            uu, vv = math.cos(yaw) * dx_ + math.sin(yaw) * dy_, -math.sin(yaw) * dx_ + math.cos(yaw) * dy_
            ce, se = math.cos(est[2]), math.sin(est[2])
            return np.c_[est[0] + ce * uu - se * vv, est[1] + se * uu + ce * vv]
        o, f = to_est(o), to_est(f)
    return o, f


def drive(sim, start, goal, fine, acts, sense, yaw0=math.pi / 2, rand=None, record=False, t_max=TIMEOUT, look=None,
          sense_every=SENSE_EVERY, vis=False, truth=None):
    """한 번 주행. sense=False면 정적 지도만 믿고 간다(대응 없음). True면 감지 카메라(120°)로 nav.Navigator가 대응한다.
    rand(random.Random)를 주면 정지 규칙 대신 무작위 속도 — 0·0.3·0.6·1배를 0.8~2.5초마다 바꾼다 (학습 데이터용).
    record=True면 5Hz로 월드모델 학습 데이터를 모아 결과의 "data"로 돌려준다:
    BEV(64×64×3)·앞 RGB(64×64)·act(직전 기록부터 이번까지 평균 명령 v, w; 첫 칸 0)·gap(돌발 물체 간격)·pose·term(충돌한 마지막 칸).
    look = nav.Tracker() (B′ 등속 외삽) 또는 wm_bev.Lookahead (C 월드모델): 5Hz로 후보 속도의 2초 충돌 위험을 보고
    nav.choose_speed로 속도를 정한다. 넓은 정지 규칙 대신 앞 0.5m 비상 정지만 남긴다. 판단에 걸린 계산 시간만큼 늦게 적용.
    차체가 랙·벽(격자 지도) 또는 돌발 물체(직사각형)에 닿으면 충돌 장면을 한 칸 더 기록하고 끝낸다."""
    world, agv, front, chase, sense_cam, ctrl, pool = sim
    loc = args.v5                                                   # v5: 추정 위치로 주행 — 실제 + 천천히 흔들리는 오차, 바닥 QR 위에서 거의 0
    lr = random.Random(f"loc-{start}-{goal}-{len(acts)}")
    sig = np.array([lr.uniform(0.02, 0.10)] * 2 + [math.radians(lr.uniform(0.3, 1.5))])   # 에피소드마다 위치 정확도가 다름
    err, tau_e, last_qr, qr_reads = np.zeros(3), 20.0, -1, 0       # 상관 20초 — 천천히 쌓이고 QR에서 한 번에 고쳐짐
    look_bev = None
    look_cfg = (getattr(look, "cams", 1), getattr(look, "bev_margin", 0.15)) if hasattr(look, "observe") else None
    zones = {}
    if rand is not None:
        zones = {"stop": -1.0, "slow": -1.0}                        # 무작위 속도: 정지 규칙 없음
    elif look is not None:
        zones = {"slow": -1.0} if args.hybrid else {"stop": 0.5, "stop_side": 0.12, "slow": -1.0}   # 앞보기: 비상 정지만 (혼합이면 규칙 정지 구역 유지)
        if args.hybrid and args.hyb_slow:                           # 혼합 + 감속 띠 (앞보기는 더 낮출 수만)
            sd, sv = map(float, args.hyb_slow.split(":"))
            zones = {"slow": sd, "slow_side": 1.2, "v_slow": sv}
    navi = nav.Navigator(fine, start, goal, robot_r=ROBOT_R, front=BODY_FRONT, half_w=BODY_HALF_W, confirm=10,
                         replan_every=20, v_max=args.vmax, backout=args.backout, back=BODY_BACK, safe_reverse=args.escape,
                         swept=args.swept, **zones)
    if look is not None:
        record = True                                              # 앞보기는 기록과 같은 BEV·행동을 쓴다
        if hasattr(look, "reset"):
            look.reset()
    pending, decisions, compute_ms, stop_steps, last_v, last_w = None, [], [], 0, 1.0, 0.0
    prev_cap, up_n = None, 0                                        # --decide score: 가속 흔들림 방지
    still_since = None                                              # --escape: 앞보기가 정지(≤0)를 고르기 시작한 시각
    state_log, look_log, acts_tr = [], [], []                       # --log-look (acts_tr: 움직이는 사람·지게차 x, y, 방향 — 30Hz)
    v_prev = w_prev = 0.0                                           # --smooth
    if navi.pp is None:
        return {"arrived": False, "contact": False, "note": "경로 없음"}, []
    plan0 = navi.pp.path[::10].round(2).tolist()
    world.reset()
    agv.set_world_pose(position=np.array([*start, 0.02]), orientation=yaw_quat(yaw0))
    if args.real_motion:                                            # 사람·지게차를 실제 같은 움직임으로 (결정적 — 조건마다 같은 움직임)
        realize_all(acts, pool, truth or fine)
    arrange(pool, acts)
    for _ in range(10):
        world.step(render=False)
    for _ in range(3):                                              # 순간이동 직후 첫 화면이 이전 장면으로 남지 않게
        world.render()
    tl, tcps = omni.timeline.get_timeline_interface(), omni.usd.get_context().get_stage().GetTimeCodesPerSecond()
    phase = [random.Random(k).uniform(0, 100) for k in range(len(acts))]
    traj, frames, min_c, min_a, max_xt, n, done, contact, hit = [], [], 9.0, 9.0, 0.0, 0, False, False, None
    hit_obj = None
    rec, cmds = {k: [] for k in ("bev", "rgb", "act", "gap", "pose")}, []
    acc, nacc, factor, f_until = np.zeros(2), 0, 1.0, 0.0
    wall0 = time.time()

    def snap(t, x, y, yaw, ca, rec_step, est):
        """돌발 물체 자세·동작 반영 → 렌더 → 감지 점 (장애물, 바닥 — 추정 좌표). rec_step이면 GIF 한 장·학습 데이터 한 칸."""
        nonlocal acc, nacc, look_bev
        stage_tc = tl.get_current_time() * tcps
        for k, a in enumerate(acts):
            e = pool[a["kind"]][a["i"]]
            if a.get("move"):
                place(e, a["x"], a["y"], a["heading"])
            if e["anim"] is not None:                               # 걷는 중이면 걷기(속도 비율로 재생), 아니면 서 있기
                name = "walk" if a.get("walking") else "idle"
                rate = a["move"].get("_v", a["move"]["speed"]) / WALK_SPEED if name == "walk" else 1.0
                set_anim(e, name, (phase[k] + t * FPS * rate) % anim_len(e, name), stage_tc)
        if args.follow_cam and chase is not None:                   # 따라가는 카메라: AGV 바로 위로 (방향은 그대로)
            chase.set_world_pose(position=np.array([x, y, args.follow_cam]))
        world.render()
        world.render()                                              # 깊이가 지금 자세의 것이 되게 두 번
        if (rec_step or (vis and args.follow_cam)) and chase is not None:   # 긴 영상은 감지마다(10Hz) 한 장
            if vis:
                rec.setdefault("view_t", []).append(round(t, 2))
            frames.append((chase.get_rgba()[:, :, :3] if args.top_cam or args.fixed_cam or args.follow_cam else
                           np.concatenate([k.get_rgba()[:, :, :3] for k in (front, chase)], axis=1)).astype(np.uint8))
        if sense or record:
            parts = [sense_points(c_, x, y, yaw, est) for c_ in [sense_cam] + EXTRA_SENSE]
            pts_f = parts[0]
            pts = (np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts]))
        else:
            pts = pts_f = (None, None)
        if record and rec_step:
            rec["bev"].append(nav.bev(fine, *est, *pts, static_margin=0.0 if args.v5 else 0.15))
            if look_cfg is not None:                                # 앞보기 모델이 학습한 BEV (v4: 앞 카메라·지도로 거름)
                look_bev = nav.bev(fine, *est, *(pts_f if look_cfg[0] == 1 else pts), static_margin=look_cfg[1])
            rec.setdefault("est", []).append(est)
            if args.v5:                                             # 깊이 원본 64×64 (나중에 원본 깊이 입력과 비교용)
                dep = np.nan_to_num(np.asarray(sense_cam.get_depth(), np.float32), nan=0.0, posinf=0.0)
                rec.setdefault("depth64", []).append(
                    np.asarray(Image.fromarray(dep).resize((64, 64), Image.BILINEAR), np.float16))
            rgb = Image.fromarray(front.get_rgba()[:, :, :3].astype(np.uint8)).resize((64, 64), Image.BILINEAR)
            rec["rgb"].append(np.asarray(rgb))
            rec["act"].append(acc / max(nacc, 1))
            rec["gap"].append(ca)
            rec["pose"].append((x, y, yaw))
            if vis:                                                 # 발표용: 감지 카메라 깊이 영상도
                rec.setdefault("depth", []).append(np.asarray(sense_cam.get_depth(), np.float32))
                if "navbase" not in rec:                            # 길찾기 지도 (처음 아는 랙·벽, 계획 칸 0.1m)
                    rec["navbase"] = {"occ": navi.base.occ.copy(), "x0": navi.base.x0, "y0": navi.base.y0, "cell": navi.base.cell}
                sig = (navi.replans, int(navi.seen.sum()))          # 확정 장애물 칸·경로가 바뀐 때만
                if sig != rec.get("_navsig"):
                    rec["_navsig"] = sig
                    rec.setdefault("navlog", []).append({"i": len(rec["pose"]) - 1, "replans": navi.replans,
                                                         "seen": np.argwhere(navi.seen).astype(np.int16),
                                                         "path": None if navi.pp is None else np.asarray(navi.pp.path, np.float32)})
            acc, nacc = np.zeros(2), 0
        return pts

    while n < t_max * FPS:
        t = n / FPS
        x, y, yaw = pose(agv)
        if not all(map(math.isfinite, (x, y, yaw))):               # 물리 계산이 터짐 — 이 주행은 버린다
            return {"arrived": False, "contact": False, "note": "물리 오류", "time_s": round(n / FPS, 1)}, frames
        traj.append((round(x, 3), round(y, 3)))
        if loc:                                                     # 위치 추정 오차 (OU 과정, 상관 3초) + 바닥 QR 보정
            err += -err / (FPS * tau_e) + sig * math.sqrt(2 / (FPS * tau_e)) * np.array([lr.gauss(0, 1) for _ in range(3)])
            if len(QR):
                kq = int(np.argmin(np.hypot(QR[:, 0] - x, QR[:, 1] - y)))
                if math.hypot(QR[kq, 0] - x, QR[kq, 1] - y) < 0.2:
                    err = np.array([lr.gauss(0, 0.01), lr.gauss(0, 0.01), lr.gauss(0, math.radians(0.3))])
                    qr_reads += kq != last_qr
                    last_qr = kq
        est = (x + err[0], y + err[1], yaw + err[2])
        for a in acts:                                              # 움직이는 돌발: AGV가 P에서 D m 안이면 출발
            m = a.get("move")
            if m is None:
                continue
            if "t0" not in m and math.dist((x, y), m["P"]) <= m["D"]:
                m["t0"] = t
            wait = False
            if m.get("yield") and "t0" in m and "tr" in m:         # --real-motion 양보: 설 거리까지 앞길 전체를 보고, 감속·가속 제한으로 서고 다시 감
                lim = MOTION["forklift" if a["kind"] == "forklift" else "person"]
                tau0 = t - m["t0"]
                along_t(m, tau0)
                v_n = max(m["_v"], 0.3)                             # 지금 원래(양보 없는) 속도
                near = False
                for s_ in np.arange(0.25, 1.0 + v_n / lim["d"] + 1e-6, 0.25):   # 0.25초 간격 (한 점만 보면 AGV를 건너뛰어 걸어 들어옴)
                    (fx, fy), fhd, _ = along_t(m, tau0 + s_)
                    if rect_gap(x, y, yaw, fx, fy, fhd, pool[a["kind"]][a["i"]]["half"]) < m.get("yield_gap", 0.5):
                        near = True
                        break
                k = m.get("k", 1.0)                                 # 원래 움직임을 따라가는 빠르기 (0 = 서 있음)
                if near:
                    k = max(0.0, k - lim["d"] / v_n / FPS)
                else:
                    k = min(1.0, k + lim["a"] / v_n / FPS)
                m["k"] = k
                m["t0"] += (1 - k) / FPS
                wait = k < 0.05
            elif m.get("yield") and "t0" in m:                     # 자연 장면: 1초 뒤 자리가 AGV에 0.5m 안이면 그 자리에서 기다림(양보)
                (fx, fy), fhd, _ = along_t(m, t - m["t0"] + 1.0)
                if rect_gap(x, y, yaw, fx, fy, fhd, pool[a["kind"]][a["i"]]["half"]) < m.get("yield_gap", 0.5):
                    m["t0"] += 1 / FPS
                    wait = True
            (a["x"], a["y"]), a["heading"], a["walking"] = along_t(m, t - m["t0"] if "t0" in m else 0.0)
            if "k" in m:
                m["_v"] *= m["k"]
            a["walking"] = a["walking"] and not wait
        if args.log_look:
            acts_tr.append([(round(a["x"], 3), round(a["y"], 3), round(a["heading"], 3)) for a in acts if a.get("move")])
        gaps = [rect_gap(x, y, yaw, a["x"], a["y"], a["heading"], pool[a["kind"]][a["i"]]["half"])
                if math.dist((x, y), (a["x"], a["y"])) < 6.0 else 9.0 for a in acts]   # 6m 밖은 닿을 리 없음
        ca = min(gaps or [9.0])
        c = min(body_clearance(truth or fine, x, y, yaw), ca)       # 충돌 판정은 실제 세계 전체 지도
        min_c, min_a = min(min_c, c), min(min_a, ca)
        if navi.pp is not None:
            max_xt = max(max_xt, navi.pp.cross_track(x, y))
        if c <= 0:
            contact, hit = True, ("hazard" if ca <= 0 else "static")
            if ca <= 0:                                             # 무엇에 부딪혔나: 종류 · 움직이던 중 · 양보하는 물체
                h_ = acts[int(np.argmin(gaps))]
                hit_obj = h_["kind"] + ("·움직임" if h_.get("walking") else "") + ("·양보자" if (h_.get("move") or {}).get("yield") else "")
            snap(t, x, y, yaw, ca, True, est)                       # 충돌 순간을 한 칸 더 (GIF·학습 데이터의 마지막)
            break
        obs = free = None
        if n % (sense_every if (sense or record) else REC_EVERY) == 0:
            obs, free = snap(t, x, y, yaw, ca, n % REC_EVERY == 0, est)
            if look is not None and n % REC_EVERY == 0 and navi.pp is not None:
                c0 = time.perf_counter()
                if isinstance(look, nav.Tracker):
                    look.update(t, navi.unconfirmed(obs) if len(obs) else obs)   # 확정된 정지 물체는 빼고 추적
                    ahead = np.r_[[est[:2]], navi.pp.path[navi.pp.k:navi.pp.k + 60]]
                    risks = look.risks(ahead, 1.0, front=BODY_FRONT, back=-BODY_BACK, half_w=BODY_HALF_W)
                else:
                    look.observe(look_bev, rec["act"][-1], rec["rgb"][-1])   # 모델이 학습한 BEV · RGB-D 모델이면 RGB도
                    kappa = last_w / last_v if last_v > 0.05 else 0.0
                    if args.pathcand:                                # 후보마다 경로를 그 속도로 2초 따라가는 행동열
                        cands = np.array([path_rollout(navi.pp, *est, sp) for sp in nav.SPEEDS], np.float32)
                        risks = look.risks(cands)
                    else:
                        risks = look.risks([(sp, kappa * sp if sp > 0 else 0.0) for sp in nav.SPEEDS])
                if args.decide == "score":
                    cap = nav.choose_speed_score(risks, args.lam)
                    if prev_cap is not None and cap > prev_cap:     # 감속은 즉시, 가속은 3번(0.6초) 연속 원할 때만
                        up_n += 1
                        cap = cap if up_n >= 3 else prev_cap
                        up_n = 0 if up_n >= 3 else up_n
                    else:
                        up_n = 0
                    prev_cap = cap
                else:
                    cap = nav.choose_speed(risks, args.tau)
                raw_cap = cap
                if args.escape:                                     # 교착 탈출: 4초 넘게 정지만 고르고 규칙 정지 구역이 비었으면 천천히 지나감
                    still_since = (still_since if still_since is not None else t) if cap <= 0 else None
                    if still_since is not None and t - still_since > 4.0 and navi.limit != 0.0:
                        cap = 0.3
                if args.log_look:
                    look_log.append({"t": round(t, 2), "risk": [round(float(q[0]), 3) for q in risks], "pick": raw_cap, "cap": cap})
                ms = (time.perf_counter() - c0) * 1000
                compute_ms.append(ms)
                pending = (n + max(1, math.ceil(ms / 1000 * FPS)), cap)   # 계산 시간만큼 늦게 적용
                if vis and getattr(look, "state", None) is not None:  # 발표용: 판단 시간을 잰 뒤라 주행에 영향 없음
                    rec.setdefault("wmlog", []).append({"i": len(rec["pose"]) - 1, "t": round(t, 2), "risk": [float(q[0]) for q in risks],
                                                        "pick": raw_cap, "cap": cap, "delay_ms": round(ms, 1),
                                                        "cands": cands if args.pathcand else None,
                                                        "state": {k_: v_.detach().float().cpu().numpy() for k_, v_ in look.state.items()}})
                decisions.append(cap)
            if pending is not None and n >= pending[0]:
                navi.cap, pending = pending[1], None
        elif look is not None and pending is not None and n >= pending[0]:
            navi.cap, pending = pending[1], None
        v, w, done = navi.step(*est, obs if sense else None, free if (sense and free is not None) else ())
        if args.debug and obs is not None and n % FPS == 0:            # 1초마다: 왜 멈추나
            un = navi.unconfirmed(obs) if len(obs) else obs
            d = np.asarray(un).reshape(-1, 2) - (x, y)
            u = math.cos(yaw) * d[:, 0] + math.sin(yaw) * d[:, 1] - BODY_FRONT
            s = -math.sin(yaw) * d[:, 0] + math.cos(yaw) * d[:, 1]
            band = (u > -0.3) & (u < navi.stop) & (np.abs(s) < BODY_HALF_W + navi.stop_side)
            print(f"[dbg] t={t:5.1f} pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):5.1f} v={v:.2f} w={w:.2f} limit={navi.limit} "
                  f"여유={navi.cmap.clearance(*est[:2]):.2f} 뒤막힘={navi.rear_block} 후진={navi.backed:.2f} 전진={navi.crept:.2f} "
                  f"경로점={len(navi.pp.path) if navi.pp is not None else 0} k={navi.pp.k if navi.pp is not None else -1} "
                  f"pp={'없음' if navi.pp is None else '있음'} obs={len(obs)} 미확정={len(un)} 정지띠={int(band.sum())} "
                  f"띠점={np.round(np.asarray(un)[band][:3], 2).tolist()}", flush=True)
        if args.log_look and n % REC_EVERY == 0:
            lim = navi.limit
            state_log.append({"t": round(t, 2), "v": round(v, 2), "w": round(w, 2), "limit": None if lim is None or lim == math.inf else lim,
                              "path": navi.pp is not None, "rear": bool(getattr(navi, "rear_block", False)), "gap": round(ca, 2),
                              "pos": [round(x, 2), round(y, 2)], "cap": navi.cap})
        if abs(v) > 0.05:                                          # 월드모델 후보의 회전량 = 최근 경로 곡률 × 속도
            last_v, last_w = abs(v), w
        if rand is not None:
            if t >= f_until:
                factor, f_until = rand.choice([0.0, 0.3, 0.6, 1.0]), t + rand.uniform(0.8, 2.5)
            if v > 0:
                v, w = v * factor, w * factor
        if done:
            break
        if args.smooth:                                             # 가속도 제한 (실제 AGV처럼 부드럽게)
            if v > v_prev:                                          # 올릴 때만 목표 속도를 천천히 따라감 (0.3초) — 경로 추종 속도 떨림 제거, 감속은 즉시
                v = v_prev + (v - v_prev) * min(1.0, 1 / (FPS * 0.3))
            v = min(max(v, v_prev - 2.0 / FPS), v_prev + 0.8 / FPS)
            w = min(max(w, w_prev - 2.0 / FPS), w_prev + 2.0 / FPS)
            v_prev, w_prev = v, w
        cmds.append((v, w))
        acc, nacc = acc + (v, w), nacc + 1
        stop_steps += int(abs(v) < 0.05)
        agv.apply_wheel_actions(ctrl.forward(command=[v, w]))
        world.step(render=args.stream)
        n += 1
        if args.stream:                                          # 1배속으로 맞춤
            lag = n / FPS - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)
    if loc and done and math.dist(traj[-1], goal) > 0.6:            # 추정으로는 도착했지만 칸 QR을 못 찾음 (리더 + 찾기 0.6m) = 칸을 놓침
        done = False
    res = {"arrived": done, "contact": contact, "hit": hit, "hit_obj": hit_obj, "goal_err": round(math.dist(traj[-1], goal), 3), "qr_reads": qr_reads,
           "time_s": round(n / FPS, 1), "min_clearance": round(min_c, 3), "min_hazard_gap": round(min_a, 3),
           "max_cross_track": round(max_xt, 3), "replans": navi.replans, "stop_s": round(stop_steps / FPS, 1),
           "plan0": plan0, "traj": traj[::10] + [traj[-1]]}
    if args.log_look:
        res.update(state_log=state_log, look_log=look_log, cmd=[[round(a, 3), round(b, 3)] for a, b in cmds], acts_tr=acts_tr)
    if look is not None:
        res.update(decide_ms=round(float(np.mean(compute_ms)), 1) if compute_ms else None,
                   slowdowns=int(sum(c < 1.0 for c in decisions)), reverses=int(sum(c < 0 for c in decisions)))
    if record and rec["bev"]:
        term = np.zeros(len(rec["bev"]), bool)
        term[-1] = contact
        res["data"] = {"bev": np.stack(rec["bev"]), "rgb": np.stack(rec["rgb"]), "act": np.array(rec["act"], np.float32),
                       "gap": np.array(rec["gap"], np.float32), "pose": np.array(rec["pose"], np.float32), "term": term,
                       "cmd": np.array(cmds, np.float32)}
        if "depth" in rec:
            res["data"]["depth"] = np.stack(rec["depth"])
        if "depth64" in rec:
            res["data"]["depth64"] = np.stack(rec["depth64"])
        res["data"]["est"] = np.array(rec["est"], np.float32)
        for k_ in ("navlog", "wmlog", "navbase", "view_t"):         # 발표용 긴 영상 (--dump-vis)
            if k_ in rec:
                res["data"][k_] = rec[k_]
    return res, frames


def path_rollout(pp, x, y, yaw, sp, steps=10, dt=0.2):
    """순수 추종으로 계획 경로를 최고 sp m/s로 steps×dt초 따라갈 때의 (v, w) 행동열 — 앞보기 후보 (sp ≤ 0이면 곧게 후진/정지)."""
    if sp <= 0:
        return [(sp, 0.0)] * steps
    q = copy.copy(pp)                                               # 지나온 위치 k만 따로 (경로 배열은 공유, 안 바뀜)
    out = []
    for _ in range(steps):
        v, w, done = q.step(x, y, yaw)
        if done:
            v, w = 0.0, 0.0
        if v > sp:
            w, v = w * sp / v, sp
        out.append((v, w))
        for _ in range(int(round(dt * FPS))):
            x, y, yaw = x + v * math.cos(yaw) / FPS, y + v * math.sin(yaw) / FPS, yaw + w / FPS
    return out


def save_gif(frames, path: Path):
    imgs = [Image.fromarray(f) for f in frames[1:]]
    if imgs:
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100, loop=0)


def load_maps(out: Path):
    """(AGV가 처음 아는 지도, 실제 세계 전체 지도). --map racks면 앞의 것은 랙·벽만 (grid_racks.npy)."""
    meta = json.loads((out / "inspect.json").read_text())
    full = nav.GridMap(np.load(out / "grid.npy"), meta["cell"], meta["x0"], meta["y0"])
    if args.map == "full":
        return full, full
    return nav.GridMap(np.load(out / "grid_racks.npy"), meta["cell"], meta["x0"], meta["y0"]), full


def run(out: Path):
    """1단계: 돌발 없이 입구 → 각 구역, 구역 → 구역."""
    fine, full = load_maps(out)
    sim = setup()
    places = {"입구": ENTRANCE, **ZONES}
    results = []
    res = out / f"nav_{args.map}"                                   # 지도 종류별 폴더 (예전 결과를 덮지 않게)
    res.mkdir(exist_ok=True)
    for a, b in ROUTES:
        tag = f"{a}_to_{b}".replace("입구", "entrance")
        r, frames = drive(sim, places[a], places[b], fine, [], sense=args.map == "racks", truth=full)
        results.append({"route": tag, **r})
        save_gif(frames, res / f"{tag}.gif")
        print(tag, {k: r.get(k) for k in ("arrived", "contact", "goal_err", "time_s", "min_clearance", "replans")}, flush=True)
    (res / "run.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    ok = [r for r in results if r["arrived"] and not r["contact"]]
    print(f"도착(접촉 없이) {len(ok)}/{len(results)}", flush=True)


def hazards(out: Path, kinds, conds):
    """돌발 상황마다 같은 배치에서 조건 비교 — none 대응 없음, respond 규칙(A·B), cv 등속 외삽 앞보기(B′),
    wm 월드모델 앞보기(C). 기준 = 충돌 없이 도착."""
    fine, full = load_maps(out)
    sim = setup(hazards=True)
    pool = sim[-1]
    wm = None
    if "wm" in conds:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import wm_bev
        wm = wm_bev.Lookahead(Path(args.wm_ckpt).expanduser())
    for kind in kinds:
        res_dir = out / ("hazards" + args.tag) / kind
        res_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for zone in list(ZONES)[:args.routes]:
            if args.zones and zone not in args.zones.split(","):
                continue
            goal = ZONES[zone]
            path = nav.plan(full.coarse(0.1), ENTRANCE, goal, robot_r=ROBOT_R, soft_r=1.5)   # 배치용 (전체 지도)
            for seed in range(args.seeds):
                if args.only and args.only != f"{zone}:{seed}":
                    continue
                rng = random.Random(f"{kind}-{zone}-{seed}")
                try:
                    acts = scenario(kind, rng, path, full, pool, ENTRANCE, goal)
                except RuntimeError as e:                               # 예: A-6 옆 통로는 잡동사니로 막혀 '랙 뒤에서 나오기'가 불가
                    print("건너뜀", kind, zone, seed, e, flush=True)
                    continue
                scen = {"kind": kind, "route": f"entrance_to_{zone}", "seed": seed, "n": len(acts),
                        "actors": [{"kind": a["kind"], "x": round(a["x"], 2), "y": round(a["y"], 2),
                                    "corners": np.round(corners(a["x"], a["y"], a["heading"], pool[a["kind"]][a["i"]]["half"]), 2).tolist(),
                                    **({"move": {k: v for k, v in a["move"].items() if k != "t0"}} if a.get("move") else {})}
                                   for a in acts]}
                for cond in conds:
                    sense = cond != "none"
                    look = {"cv": nav.Tracker(), "wm": wm}.get(cond)
                    run_acts = [dict(a, move=dict(a["move"])) if a.get("move") else dict(a) for a in acts]
                    r, frames = drive(sim, ENTRANCE, goal, fine, run_acts, sense, look=look, truth=full)
                    r.pop("data", None)
                    scen[cond] = r
                    if seed == 0 or (sense and not (r["arrived"] and not r["contact"])):
                        save_gif(frames, res_dir / f"{zone}_s{seed}_{cond}.gif")
                    print(f"{kind} {zone} s{seed} n={len(acts)} {cond}:",
                          {k: r.get(k) for k in ("arrived", "contact", "hit", "time_s", "stop_s", "min_hazard_gap", "replans",
                                                 "decide_ms", "slowdowns", "reverses") if k in r}, flush=True)
                results.append(scen)
                (res_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
        for cond in conds:
            ok = sum(s[cond]["arrived"] and not s[cond]["contact"] for s in results)
            print(f"{kind} {cond}: 충돌 없이 도착 {ok}/{len(results)}", flush=True)


def start_before(path, acts, back=10.0):
    """경로에서 처음 만나는 위험 지점(놓인 물체, 움직이는 것은 만나는 점 P)보다 back m 앞의 경로 위 점과 진행 방향."""
    d = nav.densify(path, 0.1)
    pts = [a["move"]["P"] if a.get("move") else (a["x"], a["y"]) for a in acts]
    k = min(int(np.argmin(np.hypot(d[:, 0] - px, d[:, 1] - py))) for px, py in pts)
    s = max(0, k - int(back / 0.1))
    e = min(len(d) - 1, s + 5)
    return (float(d[s][0]), float(d[s][1])), math.atan2(d[e][1] - d[s][1], d[e][0] - d[s][0])


def collect(out: Path, n_eps: int, first: int):
    """월드모델 학습 데이터. 에피소드마다 돌발 종류·구역·배치를 무작위로 정하고 위험 지점 10m 앞에서 출발해 최대 16초.
    행동 정책 4가지를 같은 비율로 섞는다 — 대응 없음(전속력, 충돌), 규칙 대응(회피), 무작위 속도 + 재계획(피하되 속도만 무작위),
    무작위 속도 + 눈 감음(여러 속도·시점에서 부딪히거나 아슬아슬하게 비껴감). 같은 장면에서 행동에 따라 결과가 갈리게 한다.
    <out>/wm_data/epNNNNN.npz + index.jsonl 한 줄."""
    fine, full = load_maps(out)
    sim = setup(hazards=True, chase_cam=False)
    pool = sim[-1]
    d = out / ("wm_data" if args.map == "full" else f"wm_data_{args.map}")   # 랙 지도 데이터는 따로
    d.mkdir(exist_ok=True)
    paths = {z: nav.plan(full.coarse(0.1), ENTRANCE, g, robot_r=ROBOT_R, soft_r=1.5) for z, g in ZONES.items()}
    with open(d / "index.jsonl", "a") as idx:
        for ep in range(first, first + n_eps):
            rng = random.Random(f"wm-{ep}")
            kind, zone = rng.choice(args.kinds.split(",")), rng.choice(sorted(ZONES))
            try:
                acts = scenario(kind, rng, paths[zone], full, pool, ENTRANCE, ZONES[zone])
            except RuntimeError:
                continue
            start, yaw0 = start_before(paths[zone], acts)
            policy = rng.choice(["none", "respond", "random", "random_blind"])
            t0 = time.time()
            r, _ = drive(sim, start, ZONES[zone], fine, acts, sense=policy in ("respond", "random"), yaw0=yaw0,
                         rand=rng if policy.startswith("random") else None, record=True, t_max=16.0,
                         sense_every=REC_EVERY, truth=full)         # 수집은 5Hz 감지 — 렌더를 절반으로
            if "data" not in r:
                continue
            np.savez_compressed(d / f"ep{ep:05d}.npz", **r["data"])
            row = {"ep": ep, "kind": kind, "zone": zone, "policy": policy, "steps": len(r["data"]["bev"]),
                   **{k: r[k] for k in ("contact", "hit", "arrived", "time_s", "min_hazard_gap")},
                   "wall_s": round(time.time() - t0, 1)}
            idx.write(json.dumps(row, ensure_ascii=False) + "\n")
            idx.flush()
            print("wm", row, flush=True)


KINDS13 = "box person crossing forklift stopgo turn forklift_turn walkstop hesitate follow oncoming forkstop emerge".split()
YIELDERS = "crossing stopgo turn hesitate walkstop oncoming emerge forklift forklift_turn forkstop".split()


def clutter(rng, path, fine, pool, start, goal, acts, p_block=0.2):
    """v5 바닥 물건 (처음 지도에 없음): 경로 위 0~3개(피해야 함) + 경로 옆 1~4m 0~4개 + 20%는 지나갈 통로 입구를 팔레트로 막음.
    반환 (물건 목록, 막은 통로 이름)."""
    used = {(a["kind"], a["i"]) for a in acts}

    def free_i(t):
        for i in range(len(pool[t])):
            if (t, i) not in used:
                used.add((t, i))
                return i
        return None
    out, types = [], BOX_TYPES + CLUTTER
    for t, x, y, yaw in sample_static(rng, path, fine, {t: (pool[t][0]["half"], 99) for t in types}, rng.randint(0, 3), start, goal):
        i = free_i(t)
        if i is not None:
            out.append({"kind": t, "i": i, "x": x, "y": y, "heading": yaw})
    d = nav.densify(path, 0.1)
    for _ in range(rng.randint(0, 4)):
        t = rng.choice(types)
        rr = math.hypot(*pool[t][0]["half"])
        for _ in range(60):
            p = d[rng.randrange(len(d))]
            a_, r_ = rng.uniform(0, 2 * math.pi), rng.uniform(1.0, 4.0)
            q = (float(p[0] + r_ * math.cos(a_)), float(p[1] + r_ * math.sin(a_)))
            if (fine.clearance(*q) >= rr + 0.05 and min(math.dist(q, start), math.dist(q, goal)) >= 2.5
                    and all(math.dist(q, (o["x"], o["y"])) > rr + 0.8 for o in out + acts)):
                i = free_i(t)
                if i is not None:
                    out.append({"kind": t, "i": i, "x": q[0], "y": q[1], "heading": rng.uniform(0, 2 * math.pi)})
                break
    blocked = None
    if AISLES and rng.random() < p_block:                           # 경로가 처음 들어가는 통로를 입구 1~1.8m 안쪽에서 가로막음
        inside = lambda al, q, m=0.0: al["rect"][0] - m <= q[0] <= al["rect"][2] + m and al["rect"][1] - m <= q[1] <= al["rect"][3] + m   # noqa: E731
        for p in d:
            if math.dist(p, start) < 3.0:
                continue
            al = next((al for al in AISLES if inside(al, p) and not inside(al, start, 0.3)), None)   # 출발한 통로는 막지 않음
            if al is None:
                continue
            m_ = min(al["mouths"], key=lambda q: math.dist(q, p))
            r_ = al["rect"]
            ax = al["axis"] == "y"
            inward = (0.0, math.copysign(1.0, (r_[1] + r_[3]) / 2 - m_[1])) if ax else (math.copysign(1.0, (r_[0] + r_[2]) / 2 - m_[0]), 0.0)
            depth = rng.uniform(1.0, 1.8)
            c_ = (m_[0] + inward[0] * (0.5 + depth), m_[1] + inward[1] * (0.5 + depth))
            if math.dist(goal, m_) < math.dist(c_, m_) + 1.0 or math.dist(c_, start) < 3.0:   # 목적지가 입구 쪽이거나 출발점 가까우면 막지 않음
                break
            width = (r_[2] - r_[0]) if ax else (r_[3] - r_[1])
            n_ = max(2, math.ceil(width / 1.15))
            for j in range(n_):
                off = -width / 2 + (j + 0.5) * width / n_
                i = free_i("pallet_load")
                if i is None:
                    break
                out.append({"kind": "pallet_load", "i": i, "x": c_[0] + (off if ax else 0.0), "y": c_[1] + (0.0 if ax else off),
                            "heading": 0.0 if ax else math.pi / 2})
            blocked = al["name"]
            break
    return out, blocked


def reachable(full, pool, acts, start, goal, robot_r=0.5):
    """실제 세계 지도 + 움직이지 않는 물체(상자·팔레트·롤 케이지·서 있는 사람)로 출발 → 목적지 길이 있나 (0.1m 칸)."""
    cm = full.coarse(0.1)
    occ = cm.occ.copy()
    for a in acts:
        if a.get("move"):
            continue
        hx, hy = pool[a["kind"]][a["i"]]["half"]
        cs = corners(a["x"], a["y"], a["heading"], (hx, hy))
        x0, y0 = np.min(cs, 0)
        x1, y1 = np.max(cs, 0)
        i0, i1 = int((y0 - cm.y0) / cm.cell), int((y1 - cm.y0) / cm.cell) + 1
        j0, j1 = int((x0 - cm.x0) / cm.cell), int((x1 - cm.x0) / cm.cell) + 1
        occ[max(i0, 0):max(i1, 0), max(j0, 0):max(j1, 0)] = True
    return nav.plan(nav.GridMap(occ, cm.cell, cm.x0, cm.y0), start, goal, robot_r=robot_r, soft_r=1.0) is not None


FAST = {"forklift", "forklift_turn", "turn", "stopgo", "emerge", "crossing"}    # v5c: 빠르거나 갑작스러운 위험 (v5b가 놓친 것) 비중 2배


def collect5(out: Path, n_eps: int, first: int):
    """v5 학습 데이터 — 실제 창고처럼 대부분 평온하고, 사람·지게차는 보통 양보하고, 가끔만 덮친다.
    에피소드: 입구(40%) 또는 칸 → 다른 통로의 칸, 최대 60초, 5Hz 기록. 장면 구성(모두 무작위):
      덮치는 위험 30% (기존 13종, 양보 안 함) · 양보하는 횡단자 50% (같은 움직임이지만 AGV 앞 0.3~1.5m에서 멈춰 기다림)
      배경 작업자 0~2명 걷기 + 0~1명 랙 앞 집품 + 지게차 50% (모두 양보) · 바닥 물건 + 20% 통로 막힘.
    주행 방식: 규칙 35% · 무작위 속도+재계획 35% · 대응 없음 15% · 무작위 속도+눈 감음 15%.
    <out>/wm_data5/epNNNNN.npz (bev·depth64·act·gap·pose·est·term·cmd·rgb) + index.jsonl 한 줄."""
    fine, full = load_maps(out)
    sim = setup(hazards=True, chase_cam=args.debug)
    pool = sim[-1]
    d = out / args.data_dir                                         # v5b: wm_data5b (근접 잘림·카메라 4대), v5c: wm_data5c (길 없는 장면 제외)
    d.mkdir(exist_ok=True)
    cm = full.coarse(0.1)
    slots = list(ENV_JSON["slots"].items())
    aisle = lambda s: s.rsplit("-", 1)[0]                           # noqa: E731
    env_name = ENV_JSON.get("seed", "simple")
    with open(d / "index.jsonl", "a") as idx:
        for ep in range(first, first + n_eps):
            rng = random.Random(f"v5-{env_name}-{ep}")
            sname, start = ("입구", tuple(ENTRANCE)) if rng.random() < 0.4 else rng.choice(slots)
            gname, goal = rng.choice([s for s in slots if aisle(s[0]) != aisle(sname)])
            start, goal = tuple(start), tuple(goal)
            path = nav.plan(cm, start, goal, robot_r=ROBOT_R, soft_r=1.5)
            if path is None or len(path) < 2:
                continue
            acts, kind, ykind = [], None, None
            if rng.random() < (0.35 if args.reach else 0.3):
                kind = rng.choices(KINDS13, [2 if k in FAST else 1 for k in KINDS13])[0] if args.reach else rng.choice(KINDS13)
                try:
                    acts += scenario(kind, rng, path, full, pool, start, goal)
                except (RuntimeError, ValueError, IndexError):
                    kind = None
            if rng.random() < 0.5:
                ykind = rng.choice(YIELDERS)
                try:
                    y_ = scenario(ykind, rng, path, full, pool, start, goal)[0]
                    y_["i"] = 1 if y_["kind"] == "forklift" else 3
                    y_["move"].update({"yield": True, "yield_gap": rng.uniform(0.3, 1.5)})
                    acts.append(y_)
                except (RuntimeError, ValueError, IndexError):
                    ykind = None
            nw, npk, nf = rng.randint(0, 2), rng.randint(0, 1), int(rng.random() < 0.5)
            try:
                acts += natural(rng, path, full, pool, start, goal, nw, npk, nf, p0=4, f0=2, yg=(0.3, 1.5), boxes=False)
            except (RuntimeError, ValueError, IndexError):
                nw = npk = nf = 0
            stuff, blocked = clutter(rng, path, full, pool, start, goal, acts)
            acts += stuff
            if args.reach and not reachable(full, pool, acts, start, goal):
                print("wm5 건너뜀", ep, "길 없음 (실제 세계 + 바닥 물건)", flush=True)   # 학습 데이터에서 뺀다 (사용자 10-09)
                continue
            policy = rng.choices(["respond", "random", "none", "random_blind"], [35, 35, 15, 15])[0]
            yaw0 = START_YAW if sname == "입구" else math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
            t0 = time.time()
            r, frames = drive(sim, start, goal, fine, acts, sense=policy in ("respond", "random"), yaw0=yaw0,
                              rand=rng if policy.startswith("random") else None, record=True, t_max=60.0,
                              sense_every=REC_EVERY, truth=full)
            if "data" not in r:
                print("wm5 건너뜀", ep, r.get("note"), flush=True)
                continue
            np.savez_compressed(d / f"ep{ep:05d}.npz", **r["data"])
            if args.debug and frames:
                save_gif(frames, d / f"ep{ep:05d}.gif")
            row = {"ep": ep, "env": env_name, "start": sname, "goal": gname, "policy": policy, "danger": kind, "yielder": ykind,
                   "walkers": nw, "pickers": npk, "forklift": nf, "clutter": len(stuff), "blocked": blocked,
                   "steps": len(r["data"]["bev"]), **{k: r[k] for k in ("contact", "hit", "arrived", "time_s", "min_hazard_gap", "qr_reads")},
                   "wall_s": round(time.time() - t0, 1)}
            idx.write(json.dumps(row, ensure_ascii=False) + "\n")
            idx.flush()
            print("wm5", row, flush=True)


def eval_scenes(out: Path):
    """v5 평가 장면 목록 (창고마다 같은 것) — 경로 6개(입구→칸 3, 칸→다른 통로 칸 3) × seed 3 × 종류 3.
    traffic: 양보하는 배경(작업자 1~2·집품 0~1·지게차 0~1) + 양보 횡단자 50% + 바닥 물건 (통로 막힘 없음)
    danger : traffic + 덮치는 위험 1 (13종을 돌아가며) / blocked: traffic + 지나갈 통로 입구를 반드시 막음."""
    env = ENV_JSON.get("seed", "simple")
    rng = random.Random(f"eval-{env}")
    slots = sorted(ENV_JSON["slots"].items())
    aisle = lambda s: s.rsplit("-", 1)[0]                           # noqa: E731
    routes, k = [], 0
    while len(routes) < 6 and k < 500:
        k += 1
        g = rng.choice(slots)
        s = ("입구", tuple(ENTRANCE)) if len(routes) < 3 else rng.choice([q for q in slots if aisle(q[0]) != aisle(g[0])])
        if (s[0], g[0]) not in [(a_[0], b_[0]) for a_, b_ in routes] and aisle(s[0]) != aisle(g[0]):
            routes.append((s, g))
    return [(ri, s, g, seed, typ) for ri, (s, g) in enumerate(routes)
            for seed in range(*map(int, args.eval_seeds.split(":"))) for typ in ("traffic", "danger", "blocked")]


def scene5(rng, typ, kind, path, full, pool, start, goal):
    """v5 시험 장면 하나: 덮치는 위험(kind, 위험 장면만) + 양보 횡단자 50% + 배경 작업자·지게차 + 바닥 물건 (막힘이면 통로를 막음).
    반환 (물체 목록, 위험 종류, 양보 종류, 막은 통로)."""
    acts, ykind = [], None
    if kind:
        try:
            acts += scenario(kind, rng, path, full, pool, start, goal)
        except (RuntimeError, ValueError, IndexError):
            kind = None
    if rng.random() < 0.5:
        ykind = rng.choice(YIELDERS)
        try:
            y_ = scenario(ykind, rng, path, full, pool, start, goal)[0]
            y_["i"] = 1 if y_["kind"] == "forklift" else 3
            y_["move"].update({"yield": True, "yield_gap": rng.uniform(0.3, 1.5)})
            acts.append(y_)
        except (RuntimeError, ValueError, IndexError):
            ykind = None
    try:
        acts += natural(rng, path, full, pool, start, goal, rng.randint(1, 2), rng.randint(0, 1), int(rng.random() < 0.5),
                        p0=4, f0=2, yg=(0.3, 1.5), boxes=False)
    except (RuntimeError, ValueError, IndexError):
        pass
    stuff, blocked = clutter(rng, path, full, pool, start, goal, acts, p_block=1.0 if typ == "blocked" else 0.0)
    return acts + stuff, kind, ykind, blocked


def demo5(out: Path, conds):
    """직접 해 보기 (10-10 사용자 요청): 창고·출발(--start, 칸 또는 '입구')·목적지(--goal)·상황(--types)·배치 번호(--demo-seed)를 정해
    무작위로 사람·지게차·바닥 물건을 뿌리고 조건마다 한 번 주행. --stream이면 맥북 Streaming Client로 실시간 (출발 전 --wait초 기다림).
    결과 <out>/demo/<이름>_<조건>.gif·.mp4."""
    fine, full = load_maps(out)
    sim = setup(hazards=True)
    pool = sim[-1]
    wm = None
    if "wm" in conds:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import wm_bev
        wm = wm_bev.Lookahead(Path(args.wm_ckpt).expanduser())
    slots = {k: tuple(v) for k, v in ENV_JSON["slots"].items()}
    for nm in (args.start, args.goal):
        if nm != "입구" and nm not in slots:
            print(f"칸 '{nm}'이 없습니다. 이 창고의 칸: {', '.join(sorted(slots))}", flush=True)
            return
    start = tuple(ENTRANCE) if args.start == "입구" else slots[args.start]
    goal = tuple(ENTRANCE) if args.goal == "입구" else slots[args.goal]
    typ = args.types or "danger"
    env = ENV_JSON.get("seed", "simple")
    rng = random.Random(f"demo-{env}-{args.start}-{args.goal}-{typ}-{args.demo_seed}")
    path = nav.plan(full.coarse(0.1), start, goal, robot_r=ROBOT_R, soft_r=1.5)
    if path is None or len(path) < 2:
        print("출발지에서 목적지까지 길이 없습니다 (다른 칸을 고르세요)", flush=True)
        return
    kind = None
    if typ == "danger":
        kind = rng.choice(KINDS13) if args.kind == "random" else args.kind
    acts, kind, ykind, blocked = scene5(rng, typ, kind, path, full, pool, start, goal)
    yaw0 = START_YAW if args.start == "입구" else math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    print(f"장면: {env} {args.start} → {args.goal} · {typ} · 덮치는 위험 {kind} · 양보 {ykind} · 막은 통로 {blocked} · 물체 {len(acts)}개", flush=True)
    res_dir = out / "demo"
    res_dir.mkdir(parents=True, exist_ok=True)
    name = f"{env}_{args.start}_{args.goal}_{typ}_{kind or 'none'}_{args.demo_seed}"   # 위험 종류까지 (같은 칸·번호라도 덮어쓰지 않게)
    for cond in conds:
        if args.stream and args.wait > 0:                           # 맥북에서 접속할 시간
            t_end, shown = time.time() + args.wait, None
            while time.time() < t_end:
                left = int(t_end - time.time())
                if left % 5 == 0 and left != shown:
                    print(f"{left}초 뒤 출발 ({'Dreamer' if cond == 'wm' else '규칙' if cond == 'respond' else '대응 없음'})", flush=True)
                    shown = left
                app.update()
                time.sleep(0.1)
        look = {"wm": wm}.get(cond)
        run_acts = [dict(a, move=dict(a["move"])) if a.get("move") else dict(a) for a in acts]
        r, frames = drive(sim, start, goal, fine, run_acts, cond != "none", yaw0=yaw0, look=look, t_max=150.0, truth=full,
                          vis=args.dump_vis, record=args.dump_vis)
        if args.dump_vis and "data" in r:                            # 발표 그림용: 0.2초마다 3인칭·앞 카메라·깊이·BEV·행동
            dd = r["data"]
            np.savez_compressed(res_dir / f"{name}_{cond}_vis.npz",
                                view=np.stack(frames) if frames else np.zeros(0), depth=dd["depth"].astype(np.float16), bev=dd["bev"],
                                act=dd["act"], gap=dd["gap"], term=dd["term"])
        word = "도착" if r.get("arrived") and not r.get("contact") else (f"충돌 ({r.get('hit_obj') or '랙'})" if r.get("contact") else "시간초과")
        print(f"결과 [{ {'wm': 'Dreamer', 'respond': '규칙', 'none': '대응 없음'}.get(cond, cond) }]: {word} · {r.get('time_s')}초 · "
              f"사람·물체와 최소 간격 {r.get('min_hazard_gap')}m", flush=True)
        if frames:
            save_gif(frames, res_dir / f"{name}_{cond}.gif")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(res_dir / f"{name}_{cond}.gif"), "-vf", "scale=iw*2:ih*2:flags=neighbor,format=yuv420p",
                            "-c:v", "libx264", "-crf", "20", str(res_dir / f"{name}_{cond}.mp4")])
            print("영상:", res_dir / f"{name}_{cond}.mp4", flush=True)


def eval5(out: Path, conds):
    """v5 평가: 처음 보는 창고에서 같은 장면을 조건마다 — none 대응 없음, respond 규칙(--vmax로 속도 대조군), wm 월드모델(--wm-ckpt).
    기준 = 목적지 칸 QR까지 충돌 없이 도착 (150초). 결과 <out>/eval5<tag>/results.json, GIF는 seed 0과 실패."""
    fine, full = load_maps(out)
    sim = setup(hazards=True)
    pool = sim[-1]
    wm = None
    if "wm" in conds:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import wm_bev
        wm = wm_bev.Lookahead(Path(args.wm_ckpt).expanduser())
    res_dir = out / ("eval5" + args.tag)
    res_dir.mkdir(parents=True, exist_ok=True)
    cm = full.coarse(0.1)
    env = ENV_JSON.get("seed", "simple")
    results = []
    for ri, (sname, start), (gname, goal), seed, typ in eval_scenes(out):
        if (args.only and args.only != f"{ri}:{seed}:{typ}") or (args.types and typ not in args.types.split(",")):
            continue
        start, goal = tuple(start), tuple(goal)
        rng = random.Random(f"e5-{env}-{ri}-{seed}-{typ}")
        path = nav.plan(cm, start, goal, robot_r=ROBOT_R, soft_r=1.5)
        if path is None:
            continue
        acts, kind, ykind, blocked = scene5(rng, typ, KINDS13[(ri * 3 + seed) % len(KINDS13)] if typ == "danger" else None,
                                            path, full, pool, start, goal)
        yaw0 = START_YAW if sname == "입구" else math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
        scen = {"env": env, "route": f"{sname}->{gname}", "ri": ri, "seed": seed, "type": typ, "danger": kind, "yielder": ykind,
                "blocked": blocked, "n_acts": len(acts)}
        for cond in conds:
            if args.rule_first and cond != "respond" and not (scen.get("respond") or {}).get("contact"):
                continue                                            # 영상 후보 찾기: 규칙이 무사하면 건너뜀
            look = {"wm": wm}.get(cond)
            run_acts = [dict(a, move=dict(a["move"])) if a.get("move") else dict(a) for a in acts]
            r, frames = drive(sim, start, goal, fine, run_acts, cond != "none", yaw0=yaw0, look=look, t_max=150.0, truth=full,
                              vis=args.dump_vis, record=args.dump_vis)
            if args.dump_vis and "data" in r:                       # 발표용 긴 영상: 주행 화면·깊이·2D 지도·상상·길찾기 지도 (make_panel_1012.py)
                with gzip.open(res_dir / f"r{ri}_s{seed}_{typ}_{cond}_vis.pkl.gz", "wb") as f_:
                    pickle.dump({**r["data"], "view": np.stack(frames) if frames else None, "env": env, "ri": ri, "seed": seed,
                                 "typ": typ, "cond": cond, "follow_cam": args.follow_cam, "start": start, "goal": goal,
                                 "cam_K": np.asarray(sim[3].get_intrinsics_matrix()) if args.follow_cam else None,   # 바닥에 경로를 겹쳐 그리려고
                                 **{k_: r.get(k_) for k_ in ("plan0", "traj", "acts_tr", "look_log", "state_log", "arrived", "contact",
                                                            "hit", "hit_obj", "time_s", "min_hazard_gap", "decide_ms")}}, f_)
            r.pop("data", None)
            scen[cond] = r
            ok = r.get("arrived") and not r.get("contact")
            if seed == 0 or (cond != "none" and not ok) or args.gif_all:
                save_gif(frames, res_dir / f"r{ri}_s{seed}_{typ}_{cond}.gif")
            print(f"eval5 {env} r{ri} s{seed} {typ} {cond}:", {k_: r.get(k_) for k_ in ("arrived", "contact", "hit", "time_s", "goal_err",
                                                                                  "replans", "slowdowns", "reverses", "note")}, flush=True)
        results.append(scen)
        (res_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    for cond in conds:
        ok = sum(bool(s.get(cond, {}).get("arrived")) and not s.get(cond, {}).get("contact") for s in results)
        print(f"eval5 {env} {cond}: 충돌 없이 도착 {ok}/{len(results)}", flush=True)


def show_episodes(out: Path, eps):
    """수집한 에피소드를 같은 난수로 다시 돌려(장면·주행 방식이 같다) 발표용 원본을 저장한다 —
    3인칭·앞 카메라 화면, 감지 카메라 깊이, BEV(학습 입력). <out>/wm_show/epNNNNN.npz (합성은 results/dreamer_safety/compose_show.py)"""
    fine, full = load_maps(out)
    sim = setup(hazards=True)
    pool = sim[-1]
    d = out / "wm_show"
    d.mkdir(exist_ok=True)
    paths = {z: nav.plan(full.coarse(0.1), ENTRANCE, g, robot_r=ROBOT_R, soft_r=1.5) for z, g in ZONES.items()}
    for ep in eps:
        kinds = "stopgo,turn,forklift_turn" if ep >= 2000 else "box,person,crossing,forklift"   # 수집 때와 같은 목록
        rng = random.Random(f"wm-{ep}")
        kind, zone = rng.choice(kinds.split(",")), rng.choice(sorted(ZONES))
        acts = scenario(kind, rng, paths[zone], full, pool, ENTRANCE, ZONES[zone])
        start, yaw0 = start_before(paths[zone], acts)
        policy = rng.choice(["none", "respond", "random", "random_blind"])
        r, frames = drive(sim, start, ZONES[zone], fine, acts, sense=policy in ("respond", "random"), yaw0=yaw0,
                          rand=rng if policy.startswith("random") else None, record=True, t_max=16.0,
                          sense_every=REC_EVERY, vis=True, truth=full)
        dd = r["data"]
        np.savez_compressed(d / f"ep{ep:05d}.npz", view=np.stack(frames), depth=dd["depth"].astype(np.float16),
                            bev=dd["bev"], gap=dd["gap"], term=dd["term"], act=dd["act"],
                            meta=np.array(json.dumps({"ep": ep, "kind": kind, "policy": policy, "zone": zone,
                                                      "contact": r["contact"], "time_s": r["time_s"]})))
        print("show", ep, kind, policy, "충돌" if r["contact"] else "", len(frames), "칸", flush=True)


if args.inspect:
    (inspect_racks if args.racks_only else inspect)(Path(args.out).expanduser())
if args.run:
    run(Path(args.out).expanduser())
if args.hazards:
    hazards(Path(args.out).expanduser(), args.hazards.split(","), args.conds.split(","))
if args.collect:
    collect(Path(args.out).expanduser(), args.collect, args.first)
if args.collect5:
    collect5(Path(args.out).expanduser(), args.collect5, args.first)
if args.eval5:
    eval5(Path(args.out).expanduser(), args.conds.split(","))
if args.show_eps:
    show_episodes(Path(args.out).expanduser(), [int(e) for e in args.show_eps.split(",")])
if args.demo:
    demo5(Path(args.out).expanduser(), args.conds.split(","))
app.close()
