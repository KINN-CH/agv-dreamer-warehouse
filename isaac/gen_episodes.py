#!/usr/bin/env python3
"""Isaac Sim 6.1 — 2×2 실험 데이터 생성기. 5090 VM에서 돌린다.

장면마다 창고 소품 하나를 AGV(idealworks iw.hub) 앞에 두고, 두 명령으로 달린다.
  · straight  1m/s 직진 — 부딪히면 그 자리에서 멈춘다
  · brake     1m/s에서 1m/s²로 감속 — 1초에 0.5m 가서 선다 (장애물은 1.0m 이상 앞)
같은 자리에서 장애물만 치운 판도 한 번씩 더 달린다 (빈 장면 대조군).

    source ~/isaacsim-venv/bin/activate
    python gen_episodes.py --scenes 3 --out ./out      # 예행연습
    python gen_episodes.py --scenes 50 --out ./out     # 본실험
    python gen_episodes.py --calib --out ./calib       # 전진 보정용 빈 장면 직진 3초 (calibrate.py)

에피소드 폴더: cond.png · future.mp4 · cond_empty.png · future_empty.mp4 · meta.json
6.1에서는 core.api·sensors.camera·wheeled_robots가 deprecated 폴더에 있어 명시적으로 켠다.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", type=int, default=3)
ap.add_argument("--first", type=int, default=0)      # 첫 장면 번호 — 학습용은 1000부터 (시험 장면 0~49와 겹치지 않게)
ap.add_argument("--out", default="./out")
ap.add_argument("--calib", action="store_true")
ap.add_argument("--cam", default="0.36,0,0.25")      # 카메라 위치 (차체 기준 x,y,z m, 수평 정면). 기본 = AGV 앞 범퍼 뒤 1인칭
ap.add_argument("--preview", default="")           # "x,y,z;x,y,z;…" — 후보 위치마다 장면 0의 첫 화면·마지막 화면만 찍고 끝
args = ap.parse_args()

app = SimulationApp({"headless": True, "width": 640, "height": 480})

from isaacsim.core.utils.extensions import enable_extension   # noqa: E402
for ext in ("isaacsim.core.api", "isaacsim.robot.wheeled_robots", "isaacsim.sensors.camera"):
    enable_extension(ext)
app.update()

import numpy as np                                                       # noqa: E402
from PIL import Image                                                    # noqa: E402
from pxr import UsdLux                                                   # noqa: E402
import omni.usd                                                          # noqa: E402
from isaacsim.core.api import World                                      # noqa: E402
from isaacsim.core.prims import SingleXFormPrim                          # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage            # noqa: E402
from isaacsim.storage.native import get_assets_root_path                # noqa: E402
from isaacsim.robot.wheeled_robots.robots import WheeledRobot            # noqa: E402
from isaacsim.robot.wheeled_robots.controllers import DifferentialController  # noqa: E402
from isaacsim.sensors.camera import Camera                              # noqa: E402

FPS, HORIZON_S, PRE_S = 30, 2.0, 0.5    # 조건 프레임 전 0.5초 직진으로 정속을 만든 뒤 2초를 기록
V, BRAKE_A = 1.0, 1.0                   # 1m/s, 정지 감속 1m/s²
OBSTACLE_GAP = (1.0, 1.6)               # 조건 프레임 때 AGV 앞면 ~ 장애물 앞면 (m). 2초 안에 직진이 닿는다
LATERAL = 0.15                          # 장애물 좌우 치우침 최대 (m)
# 창고 소품 (에셋, 반폭x, 반폭y) — 6.1 에셋 실측. 카메라(0.25m)보다 낮은 소품은 뺐다
PROPS = [("SM_CardBoxA_01", 0.350, 0.250), ("SM_CardBoxB_01", 0.250, 0.250),
         ("SM_BarelPlastic_A_01", 0.300, 0.354), ("SM_BarelPlastic_B_01", 0.300, 0.354),
         ("SM_BarelPlastic_C_01", 0.211, 0.211)]
PARK = np.array([0.0, 60.0, 0.0])       # 안 쓰는 소품을 치워두는 곳
# iw.hub 치수 (바운딩 박스 실측)
WHEEL_R, WHEEL_BASE = 0.081, 0.579
BODY_FRONT, BODY_BACK, BODY_HALF_W = 0.398, -1.033, 0.33
CAM_POS = np.array([float(v) for v in args.cam.split(",")])   # 기본 0.36,0,0.25 — 차체 앞면보다 안쪽 (앞에 두면 충돌 순간 장애물 안을 찍는다)
SPAWN = np.array([0.0, 0.0, 0.02])


def build_world():
    world = World(stage_units_in_meters=1.0, physics_dt=1 / FPS, rendering_dt=1 / FPS)
    root = get_assets_root_path()
    add_reference_to_stage(root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd", "/World/env")
    UsdLux.DomeLight.Define(omni.usd.get_context().get_stage(), "/World/dome").CreateIntensityAttr(800)
    add_reference_to_stage(root + "/Isaac/Robots/Idealworks/iwhub/iw_hub.usd", "/World/agv")
    agv = world.scene.add(WheeledRobot(prim_path="/World/agv", name="agv",
                                       wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                                       create_robot=False, position=SPAWN))
    props = []
    for i, (name, *_dims) in enumerate(PROPS):
        add_reference_to_stage(root + f"/Isaac/Environments/Simple_Warehouse/Props/{name}.usd", f"/World/obs{i}")
        props.append(SingleXFormPrim(f"/World/obs{i}", name=f"obs{i}", position=PARK))
    cam = Camera(prim_path="/World/agv/chassis/front_cam", translation=CAM_POS,
                 frequency=FPS, resolution=(640, 480))
    world.reset()
    cam.initialize()
    cam.set_focal_length(1.8)                  # 화각 약 60°
    cam.set_clipping_range(0.005, 1000.0)      # 충돌 순간 표면이 3.8cm 앞 — 기본값이면 앞면이 잘려 검게 나온다
    ctrl = DifferentialController(name="ctrl", wheel_radius=WHEEL_R, wheel_base=WHEEL_BASE)
    return world, agv, cam, props, ctrl


def clearance(agv, center, hx, hy) -> float:
    """차체(회전한 직사각형)와 장애물(축정렬 직사각형) 사이 최단 거리. 0이면 닿은 것.
    두 테두리를 2cm 간격 점으로 찍어 양방향으로 잰다."""
    def edge(x0, x1, y0, y1):
        xs, ys = np.arange(x0, x1 + 0.02, 0.02), np.arange(y0, y1 + 0.02, 0.02)
        return np.concatenate([np.c_[xs, np.full_like(xs, y0)], np.c_[xs, np.full_like(xs, y1)],
                               np.c_[np.full_like(ys, x0), ys], np.c_[np.full_like(ys, x1), ys]])

    def gap(pts, x0, x1, y0, y1):
        dx = np.maximum.reduce([x0 - pts[:, 0], np.zeros(len(pts)), pts[:, 0] - x1])
        dy = np.maximum.reduce([y0 - pts[:, 1], np.zeros(len(pts)), pts[:, 1] - y1])
        return float(np.hypot(dx, dy).min())
    p, (w, x, y, z) = agv.get_world_pose()
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    obs = (center[0] - hx, center[0] + hx, center[1] - hy, center[1] + hy)
    body = (BODY_BACK, BODY_FRONT, -BODY_HALF_W, BODY_HALF_W)
    return min(gap((edge(*obs) - p[:2]) @ rot, *body), gap(edge(*body) @ rot.T + p[:2], *obs))


def speed(cmd: str, t: float) -> float:
    return max(V - BRAKE_A * t, 0.0) if cmd == "brake" else V


def rgb(cam) -> np.ndarray:
    return cam.get_rgba()[:, :, :3].astype(np.uint8)


def drive(world, agv, cam, props, ctrl, cmd, k=None, center=None, hx=0.0, hy=0.0, start=np.zeros(3)):
    """한 번 달린다. k가 None이면 장애물을 모두 치운 빈 장면. (조건 프레임, 미래 프레임, 충돌시각, 최소거리)"""
    world.reset()
    agv.set_world_pose(position=SPAWN + start, orientation=np.array([1.0, 0, 0, 0]))
    for pr in props:
        pr.set_world_pose(position=PARK)
    if k is not None:
        props[k].set_world_pose(position=center)
    for _ in range(int(PRE_S * FPS)):
        agv.apply_wheel_actions(ctrl.forward(command=[V, 0.0]))
        world.step(render=True)
    cond, frames, ttc, min_d = rgb(cam), [], None, float("inf")
    for i in range(int(HORIZON_S * FPS)):
        # 닿으면 비상 정지 — 계속 밀면 물리 엔진이 파고듦을 허용해 카메라가 장애물 안을 찍는다
        agv.apply_wheel_actions(ctrl.forward(command=[0.0 if ttc is not None else speed(cmd, i / FPS), 0.0]))
        world.step(render=True)
        frames.append(rgb(cam))
        if k is not None:
            d = clearance(agv, center, hx, hy)
            min_d = min(min_d, d)
            if d <= 0.0 and ttc is None:
                ttc = round((i + 1) / FPS, 2)
    return cond, frames, ttc, min_d


def write_mp4(frames, path: Path) -> None:
    proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{frames[0].shape[1]}x{frames[0].shape[0]}", "-r", str(FPS), "-i", "-",
                             "-pix_fmt", "yuv420p", "-crf", "16", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


def episode(world, agv, cam, props, ctrl, sid: int, cmd: str, out: Path) -> dict:
    """같은 sid면 같은 장애물·같은 위치다."""
    rng = random.Random(sid)
    k = rng.randrange(len(PROPS))
    name, hx, hy = PROPS[k]
    gap, lateral = rng.uniform(*OBSTACLE_GAP), rng.uniform(-LATERAL, LATERAL)
    # 학습용 장면(1000~)은 출발 자리를 앞으로 1~3m, 좌우 ±0.5m 옮긴다 — 시험 장면(0~49)의 빈 화면과 겹치지 않게
    start = np.array([rng.uniform(1.0, 3.0), rng.uniform(-0.5, 0.5), 0.0]) if sid >= 1000 else np.zeros(3)
    center = np.array([V * PRE_S + BODY_FRONT + gap + hx, lateral, 0.0]) + start
    ep = out / f"ep_{sid:04d}_{cmd}"
    ep.mkdir(parents=True, exist_ok=True)
    cond, frames, ttc, min_d = drive(world, agv, cam, props, ctrl, cmd, k, center, hx, hy, start)
    Image.fromarray(cond).save(ep / "cond.png")
    write_mp4(frames, ep / "future.mp4")
    cond_e, frames_e, _, _ = drive(world, agv, cam, props, ctrl, cmd, start=start)
    Image.fromarray(cond_e).save(ep / "cond_empty.png")
    write_mp4(frames_e, ep / "future_empty.mp4")
    meta = {"scene": f"{sid:04d}", "steer": cmd, "v_mps": V,
            **({"brake_mps2": BRAKE_A} if cmd == "brake" else {}),
            "fps": FPS, "horizon_s": HORIZON_S,
            "collision": ttc is not None, "ttc_s": ttc, "min_dist_m": round(max(min_d, 0.0), 3),
            "obstacle": name, "obstacle_dist_m": round(gap, 2), "lateral_m": round(lateral, 3),
            "start_xy_m": [round(float(start[0]), 2), round(float(start[1]), 2)]}
    (ep / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"  {ep.name:20s} {name:20s} 충돌={meta['collision']!s:5s} ttc={ttc} 최소거리={meta['min_dist_m']}m",
          flush=True)
    return meta


def calib(world, agv, cam, props, ctrl, out: Path) -> None:
    """빈 장면 직진 3초 — calibrate.py가 '액션 값 → 상상 속 전진 거리'를 잴 정답."""
    global HORIZON_S
    HORIZON_S = 3.0
    ep = out / "calib_straight"
    ep.mkdir(parents=True, exist_ok=True)
    cond, frames, _, _ = drive(world, agv, cam, props, ctrl, "straight")
    Image.fromarray(cond).save(ep / "cond.png")
    write_mp4(frames, ep / "future.mp4")
    (ep / "meta.json").write_text(json.dumps({"calib": True, "v_mps": V, "fps": FPS, "measured_v_mps": 0.988}))
    print(f"  {ep} 완료 (측정 속도는 2026-09-23 실측값 0.988m/s)")


def main() -> None:
    out = Path(args.out)
    world, agv, cam, props, ctrl = build_world()
    for _ in range(10):                                 # 렌더러 예열
        world.step(render=True)
    if args.preview:
        out.mkdir(parents=True, exist_ok=True)
        rng = random.Random(0)
        k = rng.randrange(len(PROPS))
        name, hx, hy = PROPS[k]
        gap, lateral = rng.uniform(*OBSTACLE_GAP), rng.uniform(-LATERAL, LATERAL)
        center = np.array([V * PRE_S + BODY_FRONT + gap + hx, lateral, 0.0])
        for i, pos in enumerate(args.preview.split(";")):
            cam.set_local_pose(translation=np.array([float(v) for v in pos.split(",")]))
            cond, frames, ttc, _ = drive(world, agv, cam, props, ctrl, "straight", k, center, hx, hy)
            Image.fromarray(np.concatenate([cond, frames[-1]], axis=1)).save(out / f"preview_{i}_{pos}.png")
            print(f"  미리보기 {i}: 카메라 {pos} · 충돌 {ttc}s", flush=True)
    elif args.calib:
        calib(world, agv, cam, props, ctrl, out)
    else:
        rows = [episode(world, agv, cam, props, ctrl, sid, cmd, out)
                for sid in range(args.first, args.first + args.scenes) for cmd in ("straight", "brake")]
        ok = sum(1 for sid in range(args.first, args.first + args.scenes)
                 if [r["collision"] for r in rows if r["scene"] == f"{sid:04d}"] == [True, False])
        print(f"\n완료: {len(rows)} 에피소드 → {out}\n직진 충돌 · 정지 무충돌인 장면: {ok}/{args.scenes}")
    app.close()


if __name__ == "__main__":
    main()
