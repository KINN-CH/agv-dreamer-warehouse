"""애니메이션용 (10-12): FC 사람 가로지름 장면 0.2초 칸 0..K를 최종 Dreamer(v5c)에 차례로 넣고(observe), K에서 속도 5가지로 2초 상상.
주행 루프와 같은 호출(Lookahead.observe / risks). 후보는 곧은 일정 속도(회전 = 직전 곡률 × 속도). 결과 /out/wm_K.npz"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/w/src")
import torch, wm_bev
K = int(sys.argv[1]) if len(sys.argv) > 1 else 140
torch.manual_seed(0); np.random.seed(0)
z = np.load("/w/results/v5c/demo/fc_crossing_vis.npz")
la = wm_bev.Lookahead(Path("/w/results/v5c/model/wm_22000.pt"))
for k in range(K + 1):
    la.observe(z["bev"][k], z["act"][k])
v, w = z["act"][K]
kappa = w / v if v > 0.05 else 0.0
SPEEDS = (1.0, 0.6, 0.3, 0.0, -0.3)
cands = np.array([(sp, kappa * sp if sp > 0 else 0.0) for sp in SPEEDS], np.float32)
risk = np.array([r for r, _ in la.risks(cands)])
with torch.no_grad():
    wm = la.wm
    init = {k_: v_.repeat_interleave(len(SPEEDS), 0) for k_, v_ in la.state.items()}
    fut = torch.tensor(np.repeat(cands[:, None], wm_bev.H, 1) / wm_bev.ACT_SCALE, device=wm_bev.DEV)
    prior = wm.dynamics.imagine_with_action(fut, init)
    img = (wm.heads["decoder"](wm.dynamics.get_feat(prior))["image"].mode()[..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
np.savez_compressed(f"/out/wm_{K}.npz", risk=risk, imag=img, cands=cands, speeds=np.array(SPEEDS))
print("K", K, "act", z["act"][K].round(2).tolist(), "gap", round(float(z["gap"][K]), 2), "risk", (risk * 100).round(1).tolist(), "imag", img.shape)
