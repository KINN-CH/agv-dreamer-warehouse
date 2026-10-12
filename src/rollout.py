#!/usr/bin/env python3
"""조건 프레임 + 액션 → Cosmos가 상상한 미래 영상 (Cosmos 3 Nano, forward dynamics, av 도메인)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from PIL import Image

MODEL = os.environ.get("C3_MODEL", "/work/cosmos3-nano")
DOMAIN = "av"                 # 차량 ego 포즈 9차원 — AGV는 차체 자체가 움직인다 (umi는 로봇팔용)
VIEW = "ego_view"             # 전방 카메라 시점
# 프롬프트에는 조건 프레임에 보이는 것만 쓰고, 모든 조건에 같은 문장을 넣는다.
# 움직임("driving forward")이나 없는 사물("shelves")을 쓰면 상상이 그 문장을 따라갔다 (9/23).
PROMPT = ("Front camera view of a mobile robot in a large, empty indoor warehouse with "
          "white brick walls, yellow upper walls and a blue-gray concrete floor.")


def load_frame(path, size: tuple[int, int] = (480, 640)) -> Image.Image:
    arr = np.array(Image.open(path).convert("RGB"))
    return Image.fromarray(cv2.resize(arr, (size[1], size[0]), interpolation=cv2.INTER_AREA))


def save_video(frames, path, fps: int) -> None:
    w, h = frames[0].size
    cont = av.open(str(path), mode="w")
    st = cont.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": "18", "preset": "medium"}
    for f in frames:
        for pkt in st.encode(av.VideoFrame.from_ndarray(np.array(f.convert("RGB")), format="rgb24")):
            cont.mux(pkt)
    for pkt in st.encode():
        cont.mux(pkt)
    cont.close()


def build_pipeline():
    from diffusers import Cosmos3OmniPipeline
    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, enable_safety_checker=False)
    pipe.to("cuda")
    pipe.vae.enable_tiling()      # 디코드 메모리 급증을 막는다
    return pipe


def rollout(pipe, image: Image.Image, action: np.ndarray, steps: int = 20, seed: int = 0, prompt: str = PROMPT,
            guidance: float = 6.0):
    """액션 시퀀스 하나로 미래 한 편을 상상한다. (프레임들, 소요초, 피크GB).
    guidance 6은 파이프라인 기본값. 추가 학습한 모델은 1로 봐야 학습 효과가 뒤틀리지 않는다 (9/29 점검)."""
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
               generator=torch.Generator("cuda").manual_seed(seed),
               action=CosmosActionCondition(mode="forward_dynamics", chunk_size=len(action),
                                            domain_name=DOMAIN, raw_actions=torch.from_numpy(action),
                                            image=image, view_point=VIEW))
    return res.video, time.time() - t0, torch.cuda.max_memory_allocated() / 1e9


def policy(pipe, image: Image.Image, chunk: int = 24, steps: int = 20, seed: int = 0, prompt: str = PROMPT):
    """Cosmos가 직접 운전한다 — 첫 화면만 주면 행동(chunk × 9)과 그 결과 영상을 함께 내놓는다. (프레임들, 행동)."""
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition
    res = pipe(prompt=prompt, num_inference_steps=steps,
               generator=torch.Generator("cuda").manual_seed(seed),
               action=CosmosActionCondition(mode="policy", chunk_size=chunk, domain_name=DOMAIN,
                                            image=image, view_point=VIEW))
    return res.video, res.action[0].float().numpy()
