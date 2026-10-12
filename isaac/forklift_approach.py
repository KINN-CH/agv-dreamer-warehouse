#!/usr/bin/env python3
"""지게차가 서 있는 AGV로 곧장 와서 부딪히는 장면 1편 (5090 VM, Isaac Sim 6.1).

AGV는 멈춰 있고, 지게차는 등속(기본 1.5m/s)으로 계속 온다. 물리를 끄고 매 스텝 위치를 직접 옮기므로
브레이크가 없다. 가까운 면이 AGV 앞면에 닿는 순간 멈춘다(= 충돌 시각).
    python forklift_approach.py --out ~/agv/out_forklift
결과: ep_f000_straight/{cond.png, future.mp4, meta.json} — gen_episodes.py와 같은 형식 (0.5초 뒤부터 2초)
"""
import argparse
import json
import subprocess
from pathlib import Path

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="./out_forklift")
ap.add_argument("--v", type=float, default=1.5)       # 지게차 속도 m/s
ap.add_argument("--gap", type=float, default=2.5)     # 조건 프레임 때 AGV 앞면 ~ 지게차 가까운 면 (m)
args = ap.parse_args()

app = SimulationApp({"headless": True, "width": 640, "height": 480})

from isaacsim.core.utils.extensions import enable_extension   # noqa: E402
for ext in ("isaacsim.core.api", "isaacsim.robot.wheeled_robots", "isaacsim.sensors.camera"):
    enable_extension(ext)
app.update()

import numpy as np                                                       # noqa: E402
import omni.client                                                       # noqa: E402
import omni.usd                                                          # noqa: E402
from PIL import Image                                                    # noqa: E402
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics                         # noqa: E402
from isaacsim.core.api import World                                      # noqa: E402
from isaacsim.core.prims import SingleXFormPrim                          # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage            # noqa: E402
from isaacsim.storage.native import get_assets_root_path                # noqa: E402
from isaacsim.robot.wheeled_robots.robots import WheeledRobot            # noqa: E402
from isaacsim.sensors.camera import Camera                              # noqa: E402

FPS, PRE_S, HORIZON_S = 30, 0.5, 2.0
BODY_FRONT = 0.398                        # iw.hub 앞면 x (gen_episodes.py와 같음)
CAM_POS, SPAWN = np.array([0.36, 0.0, 0.25]), np.array([0.0, 0.0, 0.02])
CANDS = ["/Isaac/Props/Forklift/forklift.usd", "/Isaac/Robots/Forklift/forklift_c.usd",
         "/Isaac/Robots/Forklift/forklift_b.usd"]

root = get_assets_root_path()
asset = next((c for c in CANDS if omni.client.stat(root + c)[0] == omni.client.Result.OK), None)
print("forklift asset:", asset, flush=True)
assert asset, "지게차 에셋을 찾지 못했다"

world = World(stage_units_in_meters=1.0, physics_dt=1 / FPS, rendering_dt=1 / FPS)
stage = omni.usd.get_context().get_stage()
add_reference_to_stage(root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd", "/World/env")
UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(800)
add_reference_to_stage(root + "/Isaac/Robots/Idealworks/iwhub/iw_hub.usd", "/World/agv")
agv = world.scene.add(WheeledRobot(prim_path="/World/agv", name="agv",
                                   wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                                   create_robot=False, position=SPAWN))
add_reference_to_stage(root + asset, "/World/fork")
for p in Usd.PrimRange(stage.GetPrimAtPath("/World/fork")):   # 물리 끄기 — 모양만 남기고 위치는 직접 옮긴다
    for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.ArticulationRootAPI, UsdPhysics.CollisionAPI):
        if p.HasAPI(api):
            p.RemoveAPI(api)
fork = SingleXFormPrim("/World/fork", name="fork")
cam = Camera(prim_path="/World/agv/chassis/front_cam", translation=CAM_POS, frequency=FPS, resolution=(640, 480))
world.reset()
cam.initialize()
cam.set_focal_length(1.8)
cam.set_clipping_range(0.005, 1000.0)


def bbox():
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
        stage.GetPrimAtPath("/World/fork")).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


# 긴 축이 x(진행 방향)가 되게 돌리고, 바닥에 붙이고, 가까운 면까지의 거리를 잰다
q = np.array([1.0, 0, 0, 0])
fork.set_world_pose(position=np.array([20.0, 0, 0]), orientation=q)
lo, hi = bbox()
if hi[1] - lo[1] > hi[0] - lo[0]:
    q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    fork.set_world_pose(position=np.array([20.0, 0, 0]), orientation=q)
    lo, hi = bbox()
near, z0, cy = lo[0] - 20.0, -lo[2], -(lo[1] + hi[1]) / 2       # 원점 기준 가까운 면·바닥·좌우 중심 보정
size = hi - lo
print("forklift size (m):", np.round(size, 2), flush=True)


def place(gap):
    fork.set_world_pose(position=np.array([BODY_FRONT + gap - near, cy, z0]), orientation=q)


def rgb():
    return cam.get_rgba()[:, :, :3].astype(np.uint8)


agv.set_world_pose(position=SPAWN, orientation=np.array([1.0, 0, 0, 0]))
gap = args.gap + args.v * PRE_S
place(gap)
for _ in range(int(PRE_S * FPS)):                    # 0.5초 먼저 달려 와서 이미 움직이는 중
    gap -= args.v / FPS
    place(gap)
    world.step(render=True)
cond, frames, ttc = rgb(), [], None
for i in range(int(HORIZON_S * FPS)):
    gap = max(gap - args.v / FPS, 0.0)
    if gap == 0.0 and ttc is None:
        ttc = (i + 1) / FPS
    place(gap)
    world.step(render=True)
    frames.append(rgb())

ep = Path(args.out).expanduser() / "ep_f000_straight"
ep.mkdir(parents=True, exist_ok=True)
Image.fromarray(cond).save(ep / "cond.png")
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
                       "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(ep / "future.mp4")],
                      stdin=subprocess.PIPE)
ff.communicate(np.stack(frames).tobytes())
meta = {"asset": asset, "speed_mps": args.v, "gap_at_cond_m": args.gap, "ttc_s": ttc, "forklift_size_m": size.round(3).tolist()}
(ep / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
print("saved", ep, meta, flush=True)
app.close()
