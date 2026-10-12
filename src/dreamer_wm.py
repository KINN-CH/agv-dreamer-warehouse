#!/usr/bin/env python3
"""Dreamer(V3) 월드모델로 본실험과 같은 2×2 시험 — Cosmos와 비교 (교수님 제안, 9/30).

공개 구현 dreamerv3-torch(third_party/)의 WorldModel(RSSM + CNN 인코더·디코더)만 쓴다. 행동 학습(actor·critic)과
보상·종료 머리는 빼고 '첫 화면 + 지시 속도 → 미래 화면'만 처음부터 학습한다.
데이터는 Cosmos 추가 학습과 같다: data/train·train2의 장애물 클립(빨간 통 제외) + 옮긴 자리의 빈 장면 클립.
학습 장면 번호 20개 중 1개는 검증용으로 빼서 체크포인트를 고르는 데만 쓴다(시험 장면은 안 본다).
화면 64×64, 15fps(2프레임 간격). 한 클립 = 첫 화면 + 30장 = 2초. 명령 = 구간마다 지시 속도(m/s, 1차원).
시험: 본실험 50장면 × (장애물·빈 장면) × (직진·정지). 첫 화면에서 24스텝(1.6초)을 상상 → 실제 직진·정지 중
닮은 쪽(scoring.distance). 정답 영상도 64×64로 줄여 같은 해상도끼리 비교한다.

    agvrun src/dreamer_wm.py --cache
    agvrun src/dreamer_wm.py --train --steps=20000
    agvrun src/dreamer_wm.py --eval            # 검증 오차가 가장 낮은 체크포인트로
"""
import csv
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

DV3 = Path("/agv/third_party/dreamerv3-torch")
sys.path.insert(0, str(DV3))
sys.modules.setdefault("torch.utils.tensorboard", types.SimpleNamespace(SummaryWriter=None))   # 로그용, 안 씀
import models                                                     # noqa: E402  (dreamerv3-torch)
import experiment as E                                            # noqa: E402
import scoring as S                                               # noqa: E402

HOLDOUT = "SM_BarelPlastic_B_01"             # finetune.py와 같은 학습 제외 물체(빨간 통)
TRAIN_DIRS = [Path("/agv/data/train"), Path("/agv/data/train2")]
OUT = E.OUT / "dreamer"
RES, T, B, VAL_EVERY = 64, 31, 16, 20
DEV = "cuda:0"


def command(cmd: str, n: int) -> np.ndarray:
    """n개 구간의 지시 속도 — gen_episodes.speed와 같은 규칙(정지는 1m/s²), 구간 가운데 시각 기준."""
    t = (np.arange(n) + 0.5) / 15
    return np.ones(n) if cmd == "straight" else np.maximum(1.0 - t, 0.0)


def small(frames) -> np.ndarray:
    return np.stack([cv2.resize(np.asarray(f), (RES, RES), interpolation=cv2.INTER_AREA) for f in frames])


def load(ep: Path, cond: str, fut: str, n: int = T) -> np.ndarray:
    """첫 화면 + 2프레임 간격 n-1장 (experiment.truth와 같은 시점; n=25면 truth와 같다)."""
    from PIL import Image
    f = S.read_frames(ep / fut)
    first = np.array(Image.open(ep / cond).convert("RGB"))
    return small([first] + [f[k] for k in range(1, 2 * (n - 1), 2)])


def actions(cmd: str) -> np.ndarray:
    return np.r_[1.0, command(cmd, T - 1)].astype(np.float32)   # 0번은 is_first라 모델이 무시한다


def build_cache():
    OUT.mkdir(parents=True, exist_ok=True)
    clips = {"train": [], "val": []}
    for d in TRAIN_DIRS:
        for ep in sorted(d.glob("ep_*")):
            meta = json.loads((ep / "meta.json").read_text())
            if meta["steer"] not in E.COMMANDS:
                continue
            split = "val" if int(meta["scene"]) % VAL_EVERY == 0 else "train"
            if meta["obstacle"] != HOLDOUT:
                clips[split].append((load(ep, "cond.png", "future.mp4"), meta["steer"], "obstacle"))
            if "start_xy_m" in meta:
                clips[split].append((load(ep, "cond_empty.png", "future_empty.mp4"), meta["steer"], "empty"))
    for split, cs in clips.items():
        np.savez(OUT / f"cache_{split}.npz", img=np.stack([c[0] for c in cs]),
                 act=np.stack([actions(c[1]) for c in cs]), cmd=np.array([c[1] for c in cs]),
                 kind=np.array([c[2] for c in cs]))
        print(split, len(cs), "클립", {k: int(sum(c[2] == k for c in cs)) for k in ("obstacle", "empty")}, flush=True)


def config():
    cfg = yaml.safe_load((DV3 / "configs.yaml").read_text())["defaults"]
    for k, v in cfg.items():                      # PyYAML은 '1e-4'를 글자로 읽는다
        if isinstance(v, str):
            try:
                cfg[k] = float(v)
            except ValueError:
                pass
    cfg.update(device=DEV, compile=False, precision=32, num_actions=1, grad_heads=["decoder"],
               size=[RES, RES], batch_size=B, batch_length=T)
    return types.SimpleNamespace(**cfg)


def build():
    obs = types.SimpleNamespace(spaces={"image": types.SimpleNamespace(shape=(RES, RES, 3))})
    wm = models.WorldModel(obs, None, 0, config()).to(DEV)
    del wm.heads["reward"], wm.heads["cont"]      # 보상·종료는 배우지 않는다 — 화면 예측만
    return wm


@torch.no_grad()
def imagine(wm, first: np.ndarray, acts: np.ndarray) -> np.ndarray:
    """첫 화면(N,64,64,3) + 지시 속도(N,K) → 상상한 다음 K장 (N,K,64,64,3) uint8."""
    n = len(first)
    data = wm.preprocess({"image": first[:, None], "action": np.zeros((n, 1, 1), np.float32),
                          "is_first": np.ones((n, 1), np.float32), "is_terminal": np.zeros((n, 1), np.float32)})
    post, _ = wm.dynamics.observe(wm.encoder(data), data["action"], data["is_first"])
    init = {k: v[:, -1] for k, v in post.items()}
    prior = wm.dynamics.imagine_with_action(torch.tensor(acts[..., None], dtype=torch.float32, device=DEV), init)
    img = wm.heads["decoder"](wm.dynamics.get_feat(prior))["image"].mode()
    return (img.clamp(0, 1) * 255).round().byte().cpu().numpy()


def val_error(wm, val) -> float:
    """검증 클립: 첫 화면에서 30장을 상상해 실제와 평균제곱오차 (0~1 밝기)."""
    torch.manual_seed(0)
    pred = imagine(wm, val["img"][:, 0], val["act"][:, 1:]).astype(np.float32)
    return float(((pred - val["img"][:, 1:]) / 255.0).__pow__(2).mean())


def train(steps: int):
    tr, val = np.load(OUT / "cache_train.npz"), dict(np.load(OUT / "cache_val.npz"))
    img, act = tr["img"], tr["act"]
    wm = build()
    rng, t0, log = np.random.default_rng(0), time.time(), []
    is_first = np.zeros((B, T), np.float32)
    is_first[:, 0] = 1
    for step in range(1, steps + 1):
        idx = rng.integers(len(img), size=B)
        _, _, m = wm._train({"image": img[idx], "action": act[idx][..., None], "is_first": is_first,
                             "is_terminal": np.zeros((B, T), np.float32)})
        if step % 200 == 0:
            print(f"step {step}: 화면 손실 {float(m['image_loss']):.1f} · KL {float(m['kl']):.2f} · "
                  f"{(time.time() - t0) / 60:.0f}분", flush=True)
        if step % 2000 == 0 or step == steps:
            wm.eval()
            err = val_error(wm, val)
            wm.train()
            torch.save(wm.state_dict(), OUT / f"wm_{step}.pt")
            log.append({"step": step, "val_mse": err})
            (OUT / "train_log.json").write_text(json.dumps(log, indent=1))
            print(f"  검증 상상 오차 {err:.5f} → wm_{step}.pt", flush=True)


def evaluate(ckpt: Path | None = None):
    if ckpt is None:
        best = min(json.loads((OUT / "train_log.json").read_text()), key=lambda r: r["val_mse"])
        ckpt = OUT / f"wm_{best['step']}.pt"
    wm = build()
    wm.load_state_dict(torch.load(ckpt, map_location=DEV))
    wm.eval()
    torch.manual_seed(0)
    sids = sorted({p.name.split("_")[1] for p in E.IN.glob("ep_*_straight") if int(p.name.split("_")[1]) < 50})
    rows, vids = [], OUT / "videos"
    vids.mkdir(exist_ok=True)
    for sid in sids:
        meta = json.loads((E.IN / f"ep_{sid}_straight" / "meta.json").read_text())
        for sk, (label, cond, fut, _) in E.SCENES.items():
            refs = {c: small(E.truth(E.IN / f"ep_{sid}_{c}", cond, fut)) for c in E.COMMANDS}
            for cmd, cmd_label in E.COMMANDS.items():
                first = refs[cmd][:1]
                video = np.concatenate([first, imagine(wm, first, command(cmd, 24)[None])[0]])
                d = {c: S.distance(video, refs[c]) for c in E.COMMANDS}
                near = min(d, key=d.get)
                rows.append([sid, meta["obstacle"], label, cmd_label, E.COMMANDS[near], near == cmd,
                             round(d["straight"], 3), round(d["brake"], 3)])
                if sid in ("0000", "0001", "0003", "0007"):
                    np.save(vids / f"ep_{sid}_{cmd}{'_empty' if sk == 'empty' else ''}.npy", video)
    with open(OUT / "result.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "obstacle", "장면", "명령", "닮은 결말", "명령대로", "직진 정답과 차이", "정지 정답과 차이"])
        w.writerows(rows)

    def cell(label, cmd, pick=lambda r: True):
        rs = [r for r in rows if r[2] == label and r[3] == cmd and pick(r)]
        return f"{sum(r[5] for r in rs)}/{len(rs)}"
    held = lambda r: r[1] == HOLDOUT                                     # noqa: E731
    lines = [f"Dreamer(V3) 월드모델 — 장면 {len(sids)}개, 체크포인트 {ckpt.name}. 칸마다 '명령대로의 결말을 상상한 장면 수'",
             "", f"{'':12s}{'직진 명령':>10s}{'정지 명령':>12s}"]
    for sk, (label, *_rest) in E.SCENES.items():
        lines.append(f"{label:12s}{cell(label, '직진'):>10s}{cell(label, '정지'):>12s}")
    lines += ["", f"장애물+직진 — 학습한 물체 {cell('장애물', '직진', lambda r: not held(r))} · "
                  f"처음 보는 빨간 통 {cell('장애물', '직진', held)}",
              "비교 — Cosmos 원래 모델: 장애물+직진 0/50 · Cosmos 3차 추가 학습(장면 0~9): 3/10",
              f"Dreamer 장면 0~9: 장애물+직진 {cell('장애물', '직진', lambda r: int(r[0]) < 10)}"]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    assert command("brake", 24)[14] > 0 and command("brake", 24)[15] == 0 and command("straight", 3).tolist() == [1, 1, 1]
    opt = {a.split("=")[0]: (a.split("=")[1] if "=" in a else True) for a in sys.argv[1:]}
    if "--cache" in opt:
        build_cache()
    if "--train" in opt:
        train(int(opt.get("--steps", 20000)))
    if "--eval" in opt:
        evaluate(Path(opt["--ckpt"]) if "--ckpt" in opt else None)
