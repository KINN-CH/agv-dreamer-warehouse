#!/usr/bin/env python3
"""영상 읽기와 '두 영상이 얼마나 닮았나' — 이 실험의 유일한 기준.

Cosmos의 상상을 같은 장면의 실제 결말 두 개(직진했을 때 / 정지했을 때)와 같은 시점끼리 비교해,
더 닮은 쪽을 '상상한 결말'로 본다. 프레임을 작은 흑백(80×60)으로 줄이고 밝기를 맞춘 뒤 비교하므로
Cosmos 특유의 색감 차이보다 장면의 구조(장애물이 얼마나 커졌나, 배경이 얼마나 지나갔나)가 결정한다.

    agvrun src/scoring.py      # 자체 점검
"""
from __future__ import annotations

import numpy as np


def read_frames(path) -> np.ndarray:
    import av
    c = av.open(str(path))
    fr = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    c.close()
    return np.stack(fr)


def gray(frame) -> np.ndarray:
    import cv2
    return cv2.resize(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY), (80, 60),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def _small(frame, keep=None) -> np.ndarray:
    g = gray(frame) if keep is None else gray(frame)[keep]
    return (g - g.mean()) / (g.std() + 1e-6)


def distance(a, b, keep=None) -> float:
    """같은 시점끼리 짝지은 두 영상의 평균 차이. 작을수록 닮았다. keep(80×60)을 주면 그 픽셀만 본다."""
    return float(np.mean([((_small(x, keep) - _small(y, keep)) ** 2).mean() for x, y in zip(a, b)]))


def match_index(frames, gt_seq) -> np.ndarray:
    """프레임마다 가장 닮은 정답 프레임 번호 (calibrate.py가 상상 속 전진 거리를 잴 때 쓴다)."""
    G = np.stack([_small(f) for f in gt_seq])
    return np.array([int(np.argmin(((G - _small(f)) ** 2).mean((1, 2)))) for f in frames])


def demo() -> None:
    """자체 점검 — 색감이 달라도 '다가오는 영상'은 '다가오는 정답'에 더 닮았다고 나와야 한다."""
    T, H, W = 25, 120, 160

    def video(grow: bool) -> np.ndarray:
        v = np.full((T, H, W, 3), 200, np.uint8)
        for t in range(T):
            r = 10 + (2 * t if grow else min(t, 8))            # 다가옴: 계속 커짐 / 정지: 곧 멈춤
            v[t, H // 2 - r:H // 2 + r, W // 2 - r:W // 2 + r] = 60
        return v
    approach, stop = video(True), video(False)
    rng = np.random.default_rng(0)                             # Cosmos 렌더링처럼 색·잡음을 바꾼 '상상'
    imagined = np.clip(approach * 0.8 + 30 + rng.normal(0, 8, approach.shape), 0, 255).astype(np.uint8)
    d_a, d_s = distance(imagined, approach), distance(imagined, stop)
    print(f"  다가오는 상상 — 다가오는 정답과 차이 {d_a:.3f}, 정지 정답과 차이 {d_s:.3f}")
    assert d_a < d_s, "색감이 다르면 결말을 못 가른다"
    keep = np.ones((60, 80), bool)                            # 가운데를 가리면 가운데 차이는 안 보여야 한다
    keep[15:45, 20:60] = False
    still = np.repeat(stop[:1], T, 0)                          # 가운데 네모만 다른 영상
    edge = stop.copy()
    edge[:, :, :10] = 90                                       # 왼쪽 가장자리만 다른 영상
    d_in, d_out = distance(stop, still, keep), distance(stop, edge, keep)
    print(f"  가운데를 가리면 — 가운데만 다른 두 영상 {d_in:.3f}, 가장자리가 다른 두 영상 {d_out:.3f}")
    assert d_in < 1e-3 < d_out
    print("scoring.py 자체 점검 통과")


if __name__ == "__main__":
    demo()
