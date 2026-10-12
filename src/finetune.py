#!/usr/bin/env python3
"""Cosmos 추가 학습 — Isaac의 AGV 주행으로 '멈춰 있는 장애물'과 '충돌'을 가르친다 (LoRA).

영상을 만드는 층(add_q/k/v_proj, to_add_out)에만 작은 어댑터(LoRA)를 붙이고 나머지는 얼린다.
학습 한 번 = 실제 주행 클립(첫 프레임 + 24프레임)과 그 명령을 주고, 파이프라인이 추론 때 쓰는 입력을 그대로 만들어
'잡음 섞인 영상 → 원래 영상 방향(속도)'을 맞히게 한다 (flow matching, 추론 스케줄러와 같은 규약).

    agvrun src/finetune.py --probe      # 한 스텝이 메모리에 들어가나, 몇 초 걸리나
    agvrun src/finetune.py --overfit=/agv/data/train/ep_1009_straight   # 점검: 한 편만 외우면 따라 그리나
    agvrun src/finetune.py --train                                   # 1차: attention만, 1e-4, 1500스텝 → lora_*.pt
    agvrun src/finetune.py --train --mlp --lr=3e-4 --steps=3000 --every=1000 --tag=v2 --dirs=2
                                                                     # 2차: MLP까지, train2 포함, 빈 장면 25% → v2_*.pt
    agvrun src/finetune.py --train --mlp --lr=3e-4 --steps=2000 --every=500 --tag=v3 --dirs=2 --paired \
                           --init=/agv/results/finetune/v2_3000.pt
                                                                     # 3차: 2차에서 이어서, 같은 장면 직진·정지를 한 스텝에 짝으로

학습 장면(VM, 번호 1000~): data/train은 시험과 같은 출발 자리, data/train2는 출발 자리를 앞 1~3m·좌우 ±0.5m 옮긴 것.
원래 자리의 빈 장면은 시험 대조군과 같은 화면이라 쓰지 않는다. 빈 장면은 옮긴 자리(train2)에서 찍은 것만 쓴다.
장애물 한 종류(HOLDOUT)는 학습에서 빼서 처음 보는 물체에도 통하는지 본다.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import experiment as E       # noqa: E402
import rollout as R          # noqa: E402

TARGETS = ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")   # 영상·행동 토큰을 처리하는 attention 층
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")                   # 영상·행동 토큰 전용 MLP(mlp_moe_gen)
HOLDOUT = "SM_BarelPlastic_B_01"      # 학습에서 뺀 장애물(빨간 통) — 시험 50장면 중 11장면
TRAIN_DIRS, CACHE = [Path("/agv/data/train"), Path("/agv/data/train2")], Path("/agv/data/train_latents")
EMPTY_SHARE = 0.25                    # 학습 스텝 중 빈 장면 비율 (2차부터, 빈 장면이 있을 때)
OUT = E.OUT / "finetune"


class LoRA(torch.nn.Module):
    """y = Wx + B·A·x. B가 0에서 시작하므로 처음에는 원래 모델과 똑같다."""

    def __init__(self, base: torch.nn.Linear, rank: int = 16):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) / rank)
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


def add_lora(transformer, rank: int = 16, mlp: bool = False) -> list[torch.nn.Parameter]:
    transformer.requires_grad_(False)
    params = []
    for path, module in list(transformer.named_modules()):
        for name in TARGETS + (MLP_TARGETS if mlp and path.endswith("mlp_moe_gen") else ()):
            base = getattr(module, name, None)
            if isinstance(base, torch.nn.Linear):
                lora = LoRA(base, rank)
                setattr(module, name, lora)
                params += [lora.A, lora.B]
    return params


def save_lora(transformer, path: Path) -> None:
    torch.save({f"{n}.{k}": getattr(m, k).detach().cpu() for n, m in transformer.named_modules()
                if isinstance(m, LoRA) for k in ("A", "B")}, path)


def load_lora(transformer, path: Path) -> None:
    state = torch.load(path)
    rank = next(iter(state.values())).shape[0]
    add_lora(transformer, rank, mlp=any("mlp_moe_gen" in k for k in state))
    for n, m in transformer.named_modules():
        if isinstance(m, LoRA):
            m.A.data.copy_(state[f"{n}.A"])
            m.B.data.copy_(state[f"{n}.B"])


class _Captured(Exception):
    pass


def capture_inputs(pipe, image, action: np.ndarray) -> dict:
    """파이프라인이 첫 스텝에 모델에 넣는 입력을 그대로 가로챈다 (추론과 똑같은 포장).
    추론은 문장 있는 계산(cond)과 없는 계산(uncond)을 섞으므로(guidance 6) 둘 다 받아 둔다."""
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition
    box, original = [], pipe.transformer.forward

    def grab(*args, **kwargs):
        box.append(kwargs)
        if len(box) == 2:
            raise _Captured
        return original(*args, **kwargs)

    pipe.transformer.forward = grab
    try:
        pipe(prompt=R.PROMPT, num_inference_steps=20,
             action=CosmosActionCondition(mode="forward_dynamics", chunk_size=len(action), domain_name=R.DOMAIN,
                                          raw_actions=torch.from_numpy(action), image=image, view_point=R.VIEW))
    except _Captured:
        pass
    finally:
        pipe.transformer.forward = original
    return {"cond": box[0], "uncond": box[1]}


def clip_latents(pipe, frames: list) -> torch.Tensor:
    """실제 클립(첫 프레임 + 24프레임, PIL) → 파이프라인과 같은 방식으로 인코딩한 정답 잠재값."""
    device, dtype = pipe.transformer.device, torch.bfloat16
    with torch.no_grad():
        video, size, _, _ = pipe._prepare_action_video_conditioning(frames, 480, len(frames), device=device, dtype=dtype)
        x0 = pipe._encode_video(video)
        return pipe._remove_action_video_padding_from_latent(x0, size).float()


def train_step(pipe, packs: dict, x0: torch.Tensor, generator: torch.Generator, uncond: bool = False,
               sigma: torch.Tensor | None = None, eps: torch.Tensor | None = None) -> torch.Tensor:
    """잡음 σ를 하나 뽑아 섞고, 모델이 맞혀야 할 속도(ε − x0)와의 차이를 돌려준다. 첫 잠재 프레임은 조건이라 깨끗하게 둔다.
    짝 학습에서는 같은 σ·ε를 넘겨 받아, 직진과 정지 두 클립이 명령과 정답만 다르게 한다."""
    inputs = packs["uncond" if uncond else "cond"]
    shape = inputs["vision_tokens"][0].shape
    x0 = x0.reshape(shape)
    if sigma is None:
        sigma = torch.rand((), generator=generator, device=x0.device).clamp(0.02, 0.98)
    if eps is None:
        eps = torch.randn(x0.shape, generator=generator, device=x0.device)
    cond = torch.zeros_like(x0)
    cond[:, :, 0] = 1                                          # [B, C, T, H, W] 중 T=0
    xt = cond * x0 + (1 - cond) * ((1 - sigma) * x0 + sigma * eps)
    kw = dict(inputs)
    kw["vision_tokens"] = [xt.to(torch.bfloat16)]
    kw["vision_timesteps"] = torch.full_like(inputs["vision_timesteps"], float(sigma) * 1000)
    kw["return_dict"] = False
    preds, _, _ = pipe.transformer(**kw)
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    pred = pred.reshape(shape).float()
    target = eps - x0
    return (((pred - target) ** 2) * (1 - cond)).sum() / (1 - cond).sum()


def probe() -> None:
    pipe = R.build_pipeline()
    ep = E.IN / "ep_0000_straight"
    meta = json.loads((ep / "meta.json").read_text())
    real = E.truth(ep, "cond.png", "future.mp4")
    from PIL import Image
    frames = [Image.fromarray(f).resize((640, 480)) for f in real]
    inputs = capture_inputs(pipe, frames[0], E.command_action(meta))
    for k, v in inputs.items():
        print(k, "text tokens", tuple(v["input_ids"].shape), "first timestep", v["vision_timesteps"][:1].tolist(), flush=True)
    x0 = clip_latents(pipe, frames)
    print("clip latents", tuple(x0.shape), "vs pipeline latents", tuple(inputs["cond"]["vision_tokens"][0].shape), flush=True)
    params = add_lora(pipe.transformer)
    pipe.transformer.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(params, lr=1e-4)
    print(f"LoRA 파라미터 {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    gen = torch.Generator("cuda").manual_seed(0)
    for i in range(3):
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        with torch.enable_grad():
            loss = train_step(pipe, inputs, x0, gen, uncond=i == 2)
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        print(f"  step {i}: loss {loss.item():.4f} · {time.time() - t:.1f}s · 최대 메모리 "
              f"{torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)


def encode_all(pipe) -> dict[str, list[tuple[Path, str]]]:
    """학습 클립을 한 번만 인코딩해 둔다. {'obstacle' | 'empty': [(잠재값 파일, 명령)]}."""
    from PIL import Image
    CACHE.mkdir(parents=True, exist_ok=True)
    items = {"obstacle": [], "empty": []}
    for d in TRAIN_DIRS:
        for ep in sorted(d.glob("ep_*")):
            meta = json.loads((ep / "meta.json").read_text())
            if meta["steer"] not in E.COMMANDS:
                continue
            clips = [("obstacle", "cond.png", "future.mp4", "")] if meta["obstacle"] != HOLDOUT else []
            if "start_xy_m" in meta:          # 옮긴 자리의 빈 장면만 — 원래 자리 빈 화면은 시험 대조군과 같다
                clips.append(("empty", "cond_empty.png", "future_empty.mp4", "_empty"))
            for kind, cond, fut, suffix in clips:
                path = CACHE / f"{d.name}_{ep.name}{suffix}.pt"
                if not path.exists():
                    frames = [Image.fromarray(f).resize((640, 480)) for f in E.truth(ep, cond, fut)]
                    torch.save(clip_latents(pipe, frames).to(torch.bfloat16).cpu(), path)
                items[kind].append((path, meta["steer"]))
    return items


def train(steps: int = 1500, lr: float = 1e-4, rank: int = 16, every: int = 500, mlp: bool = False,
          tag: str = "lora", dirs: int = 1, paired: bool = False, init: str | None = None) -> None:
    """paired=True면 한 스텝에 같은 장면의 직진·정지 클립을 같은 σ·ε로 함께 학습한다 — 명령만 다를 때 결말이 갈린다는 대비."""
    global TRAIN_DIRS
    TRAIN_DIRS = TRAIN_DIRS[:dirs]
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = R.build_pipeline()
    items = encode_all(pipe)
    pairs = {k: {} for k in items}                              # 종류 → {(폴더, 장면): {명령: 파일}}
    for kind, lst in items.items():
        for path, cmd in lst:
            key = path.name.replace(f"_{cmd}", "_*")
            pairs[kind].setdefault(key, {})[cmd] = path
        pairs[kind] = [v for v in pairs[kind].values() if set(v) == set(E.COMMANDS)]
    print(f"학습 클립: 장애물 {len(items['obstacle'])}개 ({HOLDOUT} 제외) · 빈 장면 {len(items['empty'])}개 · "
          f"짝 {len(pairs['obstacle'])}+{len(pairs['empty'])} · lr {lr} · {steps}스텝 · MLP {mlp} · 짝 학습 {paired} · 시작 {init}",
          flush=True)
    from PIL import Image
    ep0 = {c: next(TRAIN_DIRS[0].glob(f"ep_*_{c}")) for c in E.COMMANDS}
    inputs = {c: capture_inputs(pipe, Image.open(ep0[c] / "cond.png").convert("RGB").resize((640, 480)),
                                E.command_action(json.loads((ep0[c] / "meta.json").read_text())))
              for c in E.COMMANDS}
    if init:
        load_lora(pipe.transformer, Path(init))
        params = [p for m in pipe.transformer.modules() if isinstance(m, LoRA) for p in (m.A, m.B)]
        for p_ in params:
            p_.requires_grad_(True)
    else:
        params = add_lora(pipe.transformer, rank, mlp)
    pipe.transformer.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(params, lr=lr)
    rng, gen = random.Random(0), torch.Generator("cuda").manual_seed(0)
    losses, t0 = [], time.time()
    for step in range(1, steps + 1):
        kind = "empty" if items["empty"] and rng.random() < EMPTY_SHARE else "obstacle"
        uncond = rng.random() < 0.1
        with torch.enable_grad():
            if paired:
                pair = rng.choice(pairs[kind])
                x0s = {c: torch.load(pair[c]).cuda().float() for c in E.COMMANDS}
                sigma = torch.rand((), generator=gen, device="cuda").clamp(0.02, 0.98)
                eps = torch.randn(x0s["straight"].shape, generator=gen, device="cuda")
                loss = sum(train_step(pipe, inputs[c], x0s[c], gen, uncond, sigma, eps) for c in E.COMMANDS) / 2
            else:
                path, cmd = rng.choice(items[kind])
                loss = train_step(pipe, inputs[cmd], torch.load(path).cuda().float(), gen, uncond)
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  step {step}: 평균 손실 {np.mean(losses[-50:]):.4f} · {(time.time() - t0) / 60:.0f}분", flush=True)
        if step % every == 0 or step == steps:
            save_lora(pipe.transformer, OUT / f"{tag}_{step}.pt")
    (OUT / f"{tag}_loss.json").write_text(json.dumps(losses))


def overfit(ep: Path, steps: int = 300, lr: float = 3e-4) -> None:
    """점검 — 충돌 영상 한 편만 반복 학습시킨 뒤 같은 첫 화면에서 그 충돌을 따라 그리나.
    따라 그리면 학습 방식은 맞고 학습량·일반화 문제, 못 그리면 학습과 추론 사이가 어긋난 것."""
    from PIL import Image
    import scoring as S
    pipe = R.build_pipeline()
    meta = json.loads((ep / "meta.json").read_text())
    real = {c: E.truth(ep.parent / ep.name.replace(meta["steer"], c), "cond.png", "future.mp4") for c in E.COMMANDS}
    frames = [Image.fromarray(f).resize((640, 480)) for f in real[meta["steer"]]]
    action = E.command_action(meta)
    inputs = capture_inputs(pipe, frames[0], action)
    x0 = clip_latents(pipe, frames)
    params = add_lora(pipe.transformer, 16, mlp=True)
    pipe.transformer.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(params, lr=lr)
    gen, losses = torch.Generator("cuda").manual_seed(0), []
    for step in range(1, steps + 1):
        with torch.enable_grad():
            loss = train_step(pipe, inputs, x0, gen, uncond=step % 10 == 0)
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  step {step}: 평균 손실 {np.mean(losses[-50:]):.4f}", flush=True)
    pipe.transformer.disable_gradient_checkpointing()
    for guidance in (6.0, 1.0):
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition
        with torch.no_grad():
            res = pipe(prompt=R.PROMPT, num_inference_steps=20, guidance_scale=guidance,
                       generator=torch.Generator("cuda").manual_seed(0),
                       action=CosmosActionCondition(mode="forward_dynamics", chunk_size=len(action), domain_name=R.DOMAIN,
                                                    raw_actions=torch.from_numpy(action), image=frames[0], view_point=R.VIEW))
        vid = OUT / f"overfit_{ep.name}_cfg{guidance:g}.mp4"
        R.save_video(res.video, vid, fps=15)
        d = {c: S.distance(S.read_frames(vid), real[c]) for c in E.COMMANDS}
        print(f"  guidance {guidance:g}: 학습한 영상({meta['steer']})과 차이 {d[meta['steer']]:.3f} · "
              f"다른 결말과 차이 {d['brake' if meta['steer'] == 'straight' else 'straight']:.3f}", flush=True)


if __name__ == "__main__":
    opt = dict(a[2:].split("=", 1) for a in sys.argv if a.startswith("--") and "=" in a)
    if "--probe" in sys.argv:
        probe()
    elif "overfit" in opt:
        overfit(Path(opt["overfit"]), steps=int(opt.get("steps", 300)))
    elif "--train" in sys.argv:
        train(steps=int(opt.get("steps", 1500)), lr=float(opt.get("lr", 1e-4)), mlp="--mlp" in sys.argv,
              tag=opt.get("tag", "lora"), every=int(opt.get("every", 500)), dirs=int(opt.get("dirs", 1)),
              paired="--paired" in sys.argv, init=opt.get("init"))
