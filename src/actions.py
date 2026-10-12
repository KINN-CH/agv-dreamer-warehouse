#!/usr/bin/env python3
"""Cosmos `av` 도메인 액션 — 이 실험은 전진만 쓴다.

av 도메인은 한 스텝 동안의 카메라 변위를 9차원으로 받는다: 이동 x·y·z(카메라 좌표계, z가 앞) + 회전 6D.
규약은 Cosmos inverse dynamics로 확인했다 (docs/01-실측기록.md ⑦). 직진·정지만 쓰므로
회전은 항등([1,0,0, 0,1,0])으로 두고 스텝마다 전진 값(z)만 바꾼다.
"""
from __future__ import annotations

import numpy as np

IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def forward(tz) -> np.ndarray:
    """스텝마다의 전진 액션 값 → (스텝 수, 9)."""
    tz = np.asarray(tz, dtype=np.float32)
    a = np.zeros((len(tz), 9), dtype=np.float32)
    a[:, 2] = tz
    a[:, 3:] = IDENTITY
    return a
