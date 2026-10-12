#!/usr/bin/env python3
"""본실험 — 2×2: 명령(직진·정지) × 장면(장애물 있음·빈 장면).

장면마다 네 칸을 Cosmos에게 상상시키고, 각 상상이 같은 장면의 실제 결말 두 개
(직진했을 때 / 정지했을 때) 중 어느 쪽과 더 닮았는지 본다 (scoring.distance).

                 직진 명령              정지 명령
  장애물 있음    실제: 부딪힌다          실제: 앞에서 선다
  빈 장면        실제: 지나간다          실제: 선다         ← 대조군. 장애물이 없을 때 명령을 따르나

    source /data/ch/agv-wm/env.sh
    agvrun src/experiment.py      # data/incoming의 모든 장면. 이미 상상한 칸은 다시 돌리지 않는다
    agvrun src/experiment.py --lora=/agv/results/finetune/lora_1500.pt [--scenes=10] [--guidance=1]
                                  # 추가 학습한 Cosmos로 같은 실험 — results/finetuned/<이름>/에 따로 쌓는다
                                  # --guidance=1: 문장 없는 계산과 섞지 않는다 (추가 학습 모델은 이걸로 본다)
    agvrun src/experiment.py --in=/agv/data/views/cam_third --view=third_person_view
                                  # 다른 카메라 위치로 찍은 장면 · Cosmos에 알려 줄 시점 — results/views/<이름>/

상상은 조건 프레임부터 1.6초(24스텝, 스텝당 1/15초). 정답도 같은 시점으로 잘라 비교한다.
결과: results/result.csv, results/summary.txt, 장면별 비교 그림 results/scene_*.png
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import actions as A          # noqa: E402
import scoring as S          # noqa: E402

IN, OUT, ROLL = Path("/agv/data/incoming"), Path("/agv/results"), Path("/agv/data/rollouts")
CALIB = OUT / "calibrate_fd.json"
LORA = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--lora=")), None)
LIMIT = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--scenes=")), None)
GUIDANCE = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--guidance=")), 6.0)
IN_DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--in=")), None)     # 다른 카메라로 찍은 장면
VIEW = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--view=")), None)     # ego_view / third_person_view
if IN_DIR or VIEW:
    IN = Path(IN_DIR) if IN_DIR else IN
    TAG = "_".join(x for x in (IN.name, VIEW or "ego_view", Path(LORA).stem if LORA else None,
                               f"g{GUIDANCE:g}" if GUIDANCE != 6.0 else None) if x)
    OUT, ROLL = OUT / "views" / TAG, ROLL.parent / "rollouts_views" / TAG
elif LORA or GUIDANCE != 6.0:
    TAG = (Path(LORA).stem if LORA else "base") + (f"_g{GUIDANCE:g}" if GUIDANCE != 6.0 else "")
    OUT, ROLL = OUT / "finetuned" / TAG, ROLL.parent / "rollouts_ft" / TAG
STRIDE, CHUNK = 2, 24          # 정답 30fps의 2프레임 = 한 스텝(1/15초), 24스텝 = 1.6초
COMMANDS = {"straight": "직진", "brake": "정지"}
SCENES = {"obstacle": ("장애물", "cond.png", "future.mp4", ""),
          "empty": ("빈 장면", "cond_empty.png", "future_empty.mp4", "_empty")}


def forward_curve():
    """보정표 → (상상 속 스텝당 전진 m, 액션 값). 포화 최고점까지만 쓰고 (0, 0)에서 시작한다."""
    tab = json.loads(CALIB.read_text())
    pts = sorted((v["action"], v["per_step"]) for k, v in tab.items() if k.startswith("calib_straight:"))
    peak = int(np.argmax([m for _, m in pts]))
    pts = sorted([(0.0, 0.0)] + [(m, a) for a, m in pts[:peak + 1]])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def command_action(meta: dict) -> np.ndarray:
    """명령 → 스텝마다의 전진 액션. 직진은 일정 속도, 정지는 일정 감속 후 0."""
    v0, acc, dt = meta["v_mps"], meta.get("brake_mps2"), STRIDE / meta["fps"]

    def pos(t):                                    # 조건 프레임 이후 t초에 간 거리
        if acc is None:
            return v0 * t
        t = min(t, v0 / acc)
        return v0 * t - acc * t * t / 2
    moved, act = forward_curve()
    return A.forward(np.interp([pos((j + 1) * dt) - pos(j * dt) for j in range(CHUNK)], moved, act))


def truth(ep: Path, cond: str, future: str) -> np.ndarray:
    """정답을 상상과 같은 시점으로 자른다 — 조건 프레임 + 2프레임 간격 24장. future의 k번째는 (k+1)/30초."""
    from PIL import Image
    fut = S.read_frames(ep / future)
    first = np.array(Image.open(ep / cond).convert("RGB"))
    return np.stack([first] + [fut[k] for k in range(STRIDE - 1, STRIDE * CHUNK, STRIDE)])


def filmstrip(sid: str, strips: dict) -> None:
    """장면 하나의 비교 그림 — 줄마다 실제/상상, 칸마다 0 · 0.27 · … · 1.6초."""
    rows = []
    for sk in SCENES:
        for cmd in COMMANDS:
            real, imag = strips[(sk, cmd)]
            rows += [(f"{sk}: real {cmd}", real), (f"{sk}: Cosmos {cmd}", imag)]
    draw_rows(rows, OUT / f"scene_{sid}.png")


def draw_rows(rows: list, path: Path) -> None:
    """(이름, 프레임 25장) 줄들을 한 장으로 — 칸마다 0 · 0.27 · … · 1.6초."""
    from PIL import Image, ImageDraw, ImageFont
    W, H, L = 160, 120, 200
    cols = range(0, CHUNK + 1, 4)
    img = Image.new("RGB", (L + W * len(cols), H * len(rows)), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for r, (label, fr) in enumerate(rows):
        draw.text((8, r * H + H // 2 - 9), label, fill="black", font=font)
        for c, k in enumerate(cols):
            img.paste(Image.fromarray(fr[k]).resize((W, H)), (L + c * W, r * H))
    img.save(path)


def summarize(rows: list[dict], n: int) -> str:
    cell = lambda s, c: sum(r["명령대로"] for r in rows if r["장면"] == s and r["명령"] == c)
    L = [f"장면 {n}개 — 칸마다 'Cosmos가 명령대로의 결말을 상상한 장면 수'", "",
         f"{'':10s}{'직진 명령':>12s}{'정지 명령':>12s}"]
    for s in ("장애물", "빈 장면"):
        L.append(f"{s:10s}{cell(s, '직진'):>9d}/{n}{cell(s, '정지'):>9d}/{n}")
    L += ["", "빈 장면은 대조군이다 — 장애물이 없을 때 Cosmos가 명령을 따르는지, 그리고 비교 기준이 작동하는지 확인한다."]
    return "\n".join(L)


def scene_pairs() -> dict[str, dict]:
    """장면 번호 → {명령: 에피소드 폴더}. 직진·정지가 모두 있고 빈 장면도 찍힌 장면만."""
    scenes: dict[str, dict] = {}
    for ep in sorted(IN.iterdir()):
        if (ep / "meta.json").exists():
            m = json.loads((ep / "meta.json").read_text())
            if m["steer"] in COMMANDS and (ep / "future_empty.mp4").exists():
                scenes.setdefault(m["scene"], {})[m["steer"]] = ep
    return {s: e for s, e in scenes.items() if set(e) == set(COMMANDS)}


def main() -> None:
    scenes = dict(list(scene_pairs().items())[:LIMIT])
    if not scenes:
        sys.exit(f"직진·정지 짝이 있는 장면이 없다: {IN}")
    print(f"장면 {len(scenes)}개 × 4칸 (명령 2 × 장면 2)" + (f" · 추가 학습 {LORA}" if LORA else "")
          + f" · guidance {GUIDANCE:g}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ROLL.mkdir(parents=True, exist_ok=True)

    pipe, rows = None, []
    for sid, eps in scenes.items():
        strips = {}
        for sk, (slabel, cond, future, suffix) in SCENES.items():
            real = {c: truth(eps[c], cond, future) for c in COMMANDS}
            for cmd, clabel in COMMANDS.items():
                ep = eps[cmd]
                meta = json.loads((ep / "meta.json").read_text())
                vid = ROLL / f"{ep.name}{suffix}.mp4"
                if not vid.exists():
                    import rollout as R
                    R.VIEW = VIEW or R.VIEW
                    if pipe is None:
                        pipe = R.build_pipeline()
                        if LORA:
                            import finetune as F
                            F.load_lora(pipe.transformer, Path(LORA))
                    fr, _, _ = R.rollout(pipe, R.load_frame(ep / cond), command_action(meta), guidance=GUIDANCE)
                    R.save_video(fr, vid, fps=round(meta["fps"] / STRIDE))
                imag = S.read_frames(vid)
                d = {c: S.distance(imag, real[c]) for c in COMMANDS}
                looks = min(d, key=d.get)
                rows.append({"scene": sid, "obstacle": meta["obstacle"], "장면": slabel, "명령": clabel,
                             "닮은 결말": COMMANDS[looks], "명령대로": looks == cmd,
                             "직진 정답과 차이": round(d["straight"], 3), "정지 정답과 차이": round(d["brake"], 3)})
                strips[(sk, cmd)] = (real[cmd], imag)
                print(f"  장면 {sid} {slabel:4s} + {clabel} 명령 → 상상이 '{COMMANDS[looks]}' 결말과 닮음 "
                      f"(차이: 직진 {d['straight']:.3f} · 정지 {d['brake']:.3f})", flush=True)
        filmstrip(sid, strips)

    with open(OUT / "result.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, len(scenes))
    (OUT / "summary.txt").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
