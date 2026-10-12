#!/usr/bin/env python3
"""배경만 비교 — 장애물 앞 직진 상상에서 카메라는 앞으로 갔나.

본실험에서 장애물 + 직진 상상에는 충돌이 없었다(0/50). 두 경우가 있다.
  (가) 카메라가 멈췄다 — 명령을 어기고 스스로 섰다
  (나) 카메라는 갔는데 장애물이 다가오지 않았다 — 장애물이 로봇과 같이 물러났다
장애물 자리를 가리고 배경만 보면 가를 수 있다. 자는 빈 장면의 실제 결말 두 개(직진 / 정지)다 —
장애물이 없으니 배경만으로 '카메라가 얼마나 갔나'를 말한다. 기준은 본실험과 같고 가린 곳만 뺀다.

  장애물 + 직진 상상   ?          직진 쪽이면 (나), 정지 쪽이면 (가)
  빈 장면 + 직진 상상  직진이어야   대조
  장애물 + 정지 상상   정지여야     대조
  장애물 + 정지 실제   정지여야     대조 — 실제로 0.5m 가서 선 영상에 이 자를 대 본다

가리는 곳: 조건 프레임에서 장애물 장면과 빈 장면이 다른 곳(장애물과 바닥 반사)을 감싸는 상자를
좌우로 반 폭씩, 위로 반 높이 넓히고 아래로는 끝까지. 상상 속 장애물이 두 배로 커져도 가려진다.

    agvrun src/background.py
결과: results/background/summary.txt, result.csv, 장애물 종류별 첫 장면 그림 scene_*.png · ghost_*.png(0초와 1.6초를 겹친 그림)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import experiment as E       # noqa: E402
import scoring as S          # noqa: E402

OUT = E.OUT / "background"


def keep_mask(ep: Path) -> np.ndarray:
    """80×60, 배경 = True."""
    g = lambda f: S.gray(np.array(Image.open(ep / f).convert("RGB")))
    ys, xs = np.nonzero(np.abs(g("cond.png") - g("cond_empty.png")) > 12)
    w, h = np.ptp(xs) + 1, np.ptp(ys) + 1
    keep = np.ones((60, 80), bool)
    keep[max(0, ys.min() - h // 2):, max(0, xs.min() - w // 2):xs.max() + 1 + w // 2] = False
    return keep


def masked(frames: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """그림용 — 가린 곳을 회색으로."""
    big = np.kron(keep, np.ones((frames.shape[1] // 60, frames.shape[2] // 80), bool))
    out = frames.copy()
    out[:, ~big] = 128
    return out


def ghost_figure(panels: list, path: Path) -> None:
    """0초와 1.6초 프레임을 반씩 겹친 그림 — 기둥이 두 겹이면 카메라가 움직인 것, 물체가 두 겹이면 커진 것."""
    from PIL import ImageDraw, ImageFont
    H, W = panels[0][1].shape[1:3]
    img = Image.new("RGB", (W * len(panels), H + 34), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for i, (label, fr) in enumerate(panels):
        img.paste(Image.fromarray(((fr[0].astype(np.float32) + fr[-1]) / 2).astype(np.uint8)), (i * W, 34))
        draw.text((i * W + 8, 6), label, fill="black", font=font)
    img.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, first_of_kind = [], set()
    for sid, eps in sorted(E.scene_pairs().items()):
        keep = keep_mask(eps["straight"])
        ruler = {c: E.truth(eps[c], "cond_empty.png", "future_empty.mp4") for c in E.COMMANDS}
        cases = {"장애물 + 직진 상상": S.read_frames(E.ROLL / f"{eps['straight'].name}.mp4"),
                 "빈 장면 + 직진 상상": S.read_frames(E.ROLL / f"{eps['straight'].name}_empty.mp4"),
                 "장애물 + 정지 상상": S.read_frames(E.ROLL / f"{eps['brake'].name}.mp4"),
                 "장애물 + 정지 실제": E.truth(eps["brake"], "cond.png", "future.mp4")}
        kind = json.loads((eps["straight"] / "meta.json").read_text())["obstacle"]
        for case, frames in cases.items():
            d = {c: S.distance(frames, ruler[c], keep) for c in E.COMMANDS}
            rows.append({"scene": sid, "obstacle": kind, "비교": case, "배경이 직진 쪽": d["straight"] < d["brake"],
                         "직진 자와 차이": round(d["straight"], 3), "정지 자와 차이": round(d["brake"], 3),
                         "배경 비율": round(keep.mean(), 2)})
        print(f"  장면 {sid} (배경 {keep.mean():.0%}) — " + " · ".join(
            f"{r['비교']}: {'직진' if r['배경이 직진 쪽'] else '정지'}" for r in rows[-len(cases):]), flush=True)
        if kind not in first_of_kind:
            first_of_kind.add(kind)
            im = cases["장애물 + 직진 상상"]
            E.draw_rows([("ruler: real straight", masked(ruler["straight"], keep)),
                         ("ruler: real brake", masked(ruler["brake"], keep)),
                         ("Cosmos straight, masked", masked(im, keep)),
                         ("Cosmos straight", im)], OUT / f"scene_{sid}.png")
            ghost_figure([("real, no obstacle: straight", ruler["straight"]),
                          ("real, no obstacle: brake", ruler["brake"]),
                          ("Cosmos, obstacle: straight", im)], OUT / f"ghost_{sid}.png")

    with open(OUT / "result.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows) // 4
    L = [f"배경만 비교 — 장면 {n}개. 칸마다 '배경이 직진 결말 쪽을 닮은 장면 수' (자 = 빈 장면의 실제 직진·정지 영상)", ""]
    for case, note in (("장애물 + 직진 상상", "← 질문. 직진 쪽이면 카메라는 갔는데 장애물이 안 다가온 것, 정지 쪽이면 카메라가 멈춘 것"),
                       ("빈 장면 + 직진 상상", "대조 — 직진 쪽이어야 정상"),
                       ("장애물 + 정지 상상", "대조 — 정지 쪽이어야 정상"),
                       ("장애물 + 정지 실제", "대조 — 정지 쪽이어야 정상")):
        L.append(f"{case:14s}{sum(r['배경이 직진 쪽'] for r in rows if r['비교'] == case):>5d}/{n}   {note}")
    summary = "\n".join(L)
    (OUT / "summary.txt").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
