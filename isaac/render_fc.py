"""FC 창고 눈으로 확인 — 지붕 숨긴 위 사진 + 안쪽 여러 지점 사진 → <out>/views/*.png"""
import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 1280, "height": 720})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import UsdGeom  # noqa: E402

out = Path(sys.argv[1]).expanduser()
(out / "views").mkdir(exist_ok=True)
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
add_reference_to_stage(str(out / "fc_warehouse.usda"), "/World/env")
# (이름, 위치, roll·pitch·yaw 도, 초점거리)
VIEWS = [("entrance_north", (-10.4, -20.5, 1.7), (0, 8, 90), 1.4),
         ("dock_west", (8.0, -19.0, 2.5), (0, 12, 165), 1.4),
         ("aisle_P2", (-24.7, 7.0, 1.6), (0, 5, 90), 1.2),
         ("aisle_R2", (-6.7, 7.0, 1.7), (0, 3, 90), 1.2),
         ("packing", (-26.5, -16.0, 2.2), (0, 14, 125), 1.4),
         ("overview", (-10.0, -20.0, 8.5), (0, 28, 90), 1.2)]
cams = []
for name, pos, rpy, f in VIEWS:
    c = Camera(prim_path=f"/World/cam_{name}", position=np.array(pos), frequency=30, resolution=(1280, 720),
               orientation=euler_angles_to_quat(np.array(rpy), degrees=True))
    cams.append((name, c, f))
top = Camera(prim_path="/World/cam_top", position=np.array([-10.0, 2.5, 60.0]), frequency=30, resolution=(1280, 1280),
             orientation=euler_angles_to_quat(np.array([0, 90, 0]), degrees=True))
world.reset()
for name, c, f in cams:
    c.initialize()
    c.set_focal_length(f)
top.initialize()
top.set_focal_length(1.3)
for _ in range(40):
    world.step(render=True)
for name, c, _ in cams:
    Image.fromarray(c.get_rgba()[:, :, :3].astype(np.uint8)).save(out / "views" / f"{name}.png")
roof = [p for p in stage.Traverse() if p.GetName().startswith("roof")]
for p in roof:
    UsdGeom.Imageable(p).MakeInvisible()
for _ in range(20):
    world.step(render=True)
Image.fromarray(top.get_rgba()[:, :, :3].astype(np.uint8)).save(out / "views" / "top.png")
print("views saved", [v[0] for v in VIEWS], "roof hidden", len(roof), flush=True)
app.close()
