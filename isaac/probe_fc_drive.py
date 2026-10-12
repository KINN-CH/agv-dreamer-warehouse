"""FC 창고에서 AGV가 왜 멈추나 — 같은 명령(전진 0.5m/s, 우회전 −0.3rad/s)을 주고 1초마다 자세·바퀴 속도를 찍는다.
인자: usd 경로 (없으면 Simple_Warehouse) · 출발 x y"""
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.robot.wheeled_robots.controllers import DifferentialController  # noqa: E402
from isaacsim.robot.wheeled_robots.robots import WheeledRobot  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

usd = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "simple" else get_assets_root_path() + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
x0, y0 = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (-10.4, -18.0)
world = World(stage_units_in_meters=1.0, physics_dt=1 / 30, rendering_dt=1 / 30)
add_reference_to_stage(usd, "/World/env")
add_reference_to_stage(get_assets_root_path() + "/Isaac/Robots/Idealworks/iwhub/iw_hub.usd", "/World/agv")
agv = world.scene.add(WheeledRobot(prim_path="/World/agv", name="agv", create_robot=False,
                                   wheel_dof_names=["left_wheel_joint", "right_wheel_joint"], position=np.array([x0, y0, 0.02])))
world.reset()
ctrl = DifferentialController(name="c", wheel_radius=0.081, wheel_base=0.579)
for n in range(30 * 8):
    agv.apply_wheel_actions(ctrl.forward(command=[0.5, -0.3 if n < 90 else 0.0]))
    world.step(render=False)
    if n % 30 == 0:
        p, q = agv.get_world_pose()
        print(f"[probe] t={n / 30:.0f}s pos={np.round(p, 3).tolist()} jv={np.round(agv.get_joint_velocities()[:4], 2).tolist()}", flush=True)
app.close()
