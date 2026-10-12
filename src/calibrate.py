#!/usr/bin/env python3
"""전진 액션 보정 — 액션 값을 바꾸면 Cosmos가 상상 속에서 실제로 얼마나 가나.

장애물 없는 보정 주행(Isaac, 직진 0.988m/s)의 첫 장면에서 전진 값을 바꿔가며 상상시키고,
상상 프레임마다 실제 주행 영상에서 가장 닮은 프레임을 찾아 '상상 속에서 간 거리'를 읽는다.
결과(results/calibrate_fd.json)를 experiment.py가 거꾸로 읽어 명령(m/s)을 액션 값으로 바꾼다.

2026-09-23 결과: 0.5 → 0.034m/스텝, 1.0 → 0.084, 2·4·8 → 0.089·0.093·0.088 (약 0.09에서 포화).

    agvrun src/calibrate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import actions as A          # noqa: E402
import rollout as R          # noqa: E402
import scoring as S          # noqa: E402

CLIP = Path("/agv/data/calib/calib_straight")
OUT = Path("/agv/results/calibrate_fd.json")
CHUNK, VALUES = 16, (0.5, 1.0, 2.0, 4.0, 8.0)


def main() -> None:
    meta = json.loads((CLIP / "meta.json").read_text())
    cond = R.load_frame(CLIP / "cond.png")
    gt_seq = [np.array(cond)] + list(S.read_frames(CLIP / "future.mp4"))   # i번째 = i/30초 뒤
    per_frame = meta["measured_v_mps"] / meta["fps"]                        # 정답 한 프레임당 전진 (m)
    tab = json.loads(OUT.read_text()) if OUT.exists() else {}
    pipe = R.build_pipeline()
    for x in VALUES:
        frames, _, _ = R.rollout(pipe, cond, A.forward([x] * CHUNK))
        idx = S.match_index(frames, gt_seq)
        moved = float(np.polyfit(np.arange(len(idx)), idx, 1)[0]) * per_frame
        tab[f"calib_straight:{x}"] = {"action": x, "matched": idx.tolist(), "per_step": round(moved, 4)}
        print(f"  전진 액션 {x:4.1f} → 상상 속 {moved:.4f} m/스텝", flush=True)
    OUT.write_text(json.dumps(tab, indent=1))


if __name__ == "__main__":
    main()
