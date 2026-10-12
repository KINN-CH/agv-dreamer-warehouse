#!/usr/bin/env python3
"""물류창고 안전 월드모델 — AGV 센서로 만든 BEV + 행동(v, w)으로 앞으로 2초를 상상하고 충돌 확률을 낸다.

우리 학습의 요점: 범용 영상 모델(Cosmos)은 충돌을 본 적이 없어 충돌을 못 그렸다(0/50, 낙관 편향). 그래서 Isaac 시험장에서
충돌·아슬아슬한 장면을 일부러 많이 넣은 데이터(isaac/logistics.py --collect)로 처음부터 학습하고, 충돌을 '에피소드 종료'로
넣어 Dreamer(dreamerv3-torch WorldModel)의 종료 예측 머리(cont)가 곧 충돌 확률 머리가 되게 한다.

데이터: <data>/epNNNNN.npz — bev (L,64,64,3) uint8 (R 정적 지도, G 본 장애물, B 보인 바닥; AGV 중심·진행 방향 위),
        act (L,2) [v m/s, w rad/s; act[t] = t−1→t 구간 평균 명령, act[0] = 0], term (L,) 충돌한 마지막 칸. 5Hz.
검증: 에피소드 번호 % 10 == 0 (학습에 안 씀).
평가: 검증 에피소드의 각 시점 t에서 실제 다음 H칸(2초) 행동으로 상상 → 충돌 확률 = 1 − Π cont.
      정답 = 실제로 H칸 안에 충돌. AUROC + 같은 시점에 '멈춤'을 넣으면 확률이 내려가는지.
      비교 기준: '앞 통로의 가장 가까운 장애물 거리' (BEV G 채널만 보는 단순 규칙).

    python src/wm_bev.py --train --data ~/agv/logistics/wm_data --out ~/agv/wm_bev --steps 20000
    python src/wm_bev.py --eval  --data ~/agv/logistics/wm_data --out ~/agv/wm_bev
    python src/wm_bev.py --selftest
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import yaml

DV3 = Path(__file__).resolve().parent.parent / "third_party" / "dreamerv3-torch"
sys.path.insert(0, str(DV3))
sys.modules.setdefault("torch.utils.tensorboard", types.SimpleNamespace(SummaryWriter=None))   # 로그용, 안 씀
import models                                                     # noqa: E402  (dreamerv3-torch)

RES, T, B, H = 64, 24, 16, 10            # 화면, 학습 구간(4.8초), 배치, 상상 길이(2초)
CONT_SCALE, TERM_FRAC = 5.0, 0.5          # 충돌(종료) 손실 가중치, 충돌로 끝나는 구간을 뽑는 비율 (--cont-scale, --term-frac)
CHAN_OFF = ""                             # 끄는 BEV 채널 (--chan-off, 예 "R" 또는 "RB") — 입력·복원 모두 0으로
IMG_SCALE = 1.0                           # 화면 복원 손실 가중치 (--img-scale) — 꽉 찬 v5 BEV는 화면 손실이 v4의 10~30배라 충돌 신호를 묻는다
VAL_MAX = 10 ** 9                         # 학습 중 검증 에피소드 수 상한 (--val-max; v5 = 100)
CAMS, BEV_MARGIN = 1, 0.15                 # 데이터의 깊이 카메라 수, BEV 초록을 지운 지도 둘레 (--cams, --bev-margin; v5 = 3, 0)
CH = 3                                    # 입력 채널: 3 = BEV(깊이)만, 6 = BEV + 앞 카메라 RGB (--rgb, RGB-D)
ACT_SCALE = np.array([1.0, 1.2], np.float32)   # v, w를 대략 −1~1로
BEV_ROW0, BEV_CELL = 44, 0.15            # isaac/nav.py와 같음 — AGV 기준점 줄, 칸 크기
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def load(data: Path):
    eps = {"train": [], "val": []}
    for f in sorted(data.glob("ep*.npz")):
        try:                                      # 수집 중이면 마지막 파일이 덜 써졌을 수 있다
            z = np.load(f)
            e = {"ep": int(f.stem[2:]), "bev": z["bev"], "act": (z["act"] / ACT_SCALE).astype(np.float32),
                 "term": z["term"].astype(np.float32)}
            e["img"] = np.concatenate([z["bev"], z["rgb"]], -1) if CH == 6 else chan_off(z["bev"], CHAN_OFF)   # 모델 입력
        except Exception:                         # noqa: BLE001
            continue
        eps["val" if e["ep"] % 10 == 0 else "train"].append(e)
    return eps


def chan_off(bev, off):
    """BEV 채널 끄기 (R 지도·G 장애물·B 바닥 중 off에 든 것을 0으로)."""
    if not off:
        return bev
    bev = np.array(bev, copy=True)
    for k, c in enumerate("RGB"):
        if c in off:
            bev[..., k] = 0
    return bev


def batch(eps, rng):
    """길이 T 구간 B개. 충돌 에피소드는 절반 확률로 충돌로 끝나는 구간을 뽑는다 (충돌 칸이 드물어서)."""
    pool = [e for e in eps if len(e["bev"]) >= T]
    img = np.zeros((B, T, RES, RES, CH), np.uint8)
    act, term = np.zeros((B, T, 2), np.float32), np.zeros((B, T), np.float32)
    for b in range(B):
        e = pool[rng.integers(len(pool))]
        L = len(e["bev"])
        s = L - T if e["term"][-1] and rng.random() < TERM_FRAC else int(rng.integers(0, L - T + 1))
        img[b], act[b], term[b] = e["img"][s:s + T], e["act"][s:s + T], e["term"][s:s + T]
    is_first = np.zeros((B, T), np.float32)
    is_first[:, 0] = 1
    return {"image": img, "action": act, "is_first": is_first, "is_terminal": term}


def config():
    cfg = yaml.safe_load((DV3 / "configs.yaml").read_text())["defaults"]
    for k, v in cfg.items():                      # PyYAML은 '1e-4'를 글자로 읽는다
        if isinstance(v, str):
            try:
                cfg[k] = float(v)
            except ValueError:
                pass
    cfg.update(device=DEV, compile=False, precision=32, num_actions=2, grad_heads=["decoder", "cont"],
               size=[RES, RES], batch_size=B, batch_length=T)
    cfg["cont_head"] = dict(cfg["cont_head"], loss_scale=CONT_SCALE)   # 충돌 칸이 드물어 종료 손실을 키운다
    return types.SimpleNamespace(**cfg)


def build(ch=None):
    obs = types.SimpleNamespace(spaces={"image": types.SimpleNamespace(shape=(RES, RES, ch or CH))})
    wm = models.WorldModel(obs, None, 0, config()).to(DEV)
    del wm.heads["reward"]                        # 보상은 없다 — 화면 예측 + 충돌(종료) 예측만
    wm._scales["image"] = IMG_SCALE
    return wm


@torch.no_grad()
def episode_risk(wm, e, fut=None):
    """에피소드 e의 각 시점 t에서 다음 H칸 행동(fut 없으면 실제 행동, 모자라면 마지막 행동 반복)으로 상상.
    반환 (L, H) 칸별 충돌 확률, (L, H, 64, 64, 3) 상상 화면은 따로 imagine_frames로."""
    L = len(e["bev"])
    data = wm.preprocess({"image": e["img"][None], "action": e["act"][None],
                          "is_first": np.r_[1.0, np.zeros(L - 1)][None].astype(np.float32),
                          "is_terminal": np.zeros((1, L), np.float32)})
    post, _ = wm.dynamics.observe(wm.encoder(data), data["action"], data["is_first"])
    init = {k: v[0] for k, v in post.items()}                     # (L, ...) — 시점마다 하나씩, 배치로 상상
    if fut is None:
        pad = np.concatenate([e["act"], np.repeat(e["act"][-1:], H, 0)])
        fut = np.stack([pad[t + 1:t + 1 + H] for t in range(L)])
    prior = wm.dynamics.imagine_with_action(torch.tensor(fut, dtype=torch.float32, device=DEV), init)
    cont = wm.heads["cont"](wm.dynamics.get_feat(prior)).mean.squeeze(-1)
    return (1 - cont).cpu().numpy(), prior


def p_hit(p):
    """칸별 충돌 확률 (…, H) → H칸 안에 한 번이라도 충돌할 확률."""
    return 1 - np.prod(1 - p, axis=-1)


def near_score(bev):
    """비교 기준: 앞 통로(좌우 0.8m, 앞 0~4m)에서 가장 가까운 장애물(G) 거리 → 가까울수록 높은 점수."""
    fwd = (BEV_ROW0 - np.arange(RES)[:, None] - 0.5) * BEV_CELL
    left = (RES / 2 - np.arange(RES)[None, :] - 0.5) * BEV_CELL
    zone = (fwd > 0) & (fwd < 4) & (np.abs(left) < 0.8)
    out = []
    for b in bev:
        d = np.hypot(fwd, left)[zone & (b[..., 1] > 0)]
        out.append(-(d.min() if len(d) else 9.0))
    return np.array(out)


def auroc(score, label):
    """순위로 계산한 AUROC (동점 무시)."""
    r = np.empty(len(score))
    r[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    pos = label.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def platt(score, label):
    """1차원 로지스틱 보정 P(충돌) = σ(a·logit(p) + b) — 충돌을 강조해 배운 확률이 높게 나오는 것을 검증 정답에 맞춘다."""
    p = np.clip(np.asarray(score, float), 1e-4, 1 - 1e-4)
    x, y, ab = np.log(p / (1 - p)), np.asarray(label, float), np.array([1.0, 0.0])

    def nll(c):
        z = c[0] * x + c[1]
        return float((np.logaddexp(0, z) - y * z).sum())
    for _ in range(100):                                           # 뉴턴법 + 보폭 줄이기 (그냥 뉴턴은 진동·발산한다)
        q = 0.5 * (1 + np.tanh(0.5 * (ab[0] * x + ab[1])))
        w = q * (1 - q) + 1e-9
        g = np.array([((q - y) * x).sum(), (q - y).sum()])
        hm = np.array([[(w * x * x).sum(), (w * x).sum()], [(w * x).sum(), w.sum()]])
        step, t, f0 = np.linalg.solve(hm + 1e-6 * np.eye(2), g), 1.0, nll(ab)
        while nll(ab - t * step) > f0 and t > 1e-6:
            t /= 2
        ab = ab - t * step
        if np.abs(t * step).max() < 1e-8:
            break
    return [float(ab[0]), float(ab[1])]


def calibrated(p, calib):
    if calib is None:
        return np.asarray(p)
    q = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return 1 / (1 + np.exp(-(calib[0] * np.log(q / (1 - q)) + calib[1])))


def evaluate(wm, val):
    """검증 에피소드 전체: 실제 행동 상상 AUROC, 단순 기준 AUROC, 충돌 직전(2초 안) 시점에서 실제 vs 멈춤 확률."""
    scores, near, labels, p_act, p_stop = [], [], [], [], []
    for e in val:
        L, hit = len(e["bev"]), bool(e["term"][-1])
        p, _ = episode_risk(wm, e)
        ps, _ = episode_risk(wm, e, np.zeros((L, H, 2), np.float32))
        nz = near_score(e["bev"])
        for t in range(1, L - 1):
            if hit:
                lab = (L - 1 - t) <= H
            elif t <= L - 1 - H:
                lab = False
            else:
                continue                                          # 미래가 잘려 정답을 모름
            scores.append(p_hit(p[t]))
            near.append(nz[t])
            labels.append(lab)
            if lab:
                p_act.append(p_hit(p[t]))
                p_stop.append(p_hit(ps[t]))
    labels = np.array(labels)
    calib = platt(scores, labels) if labels.any() and (~labels).any() else None
    cs = calibrated(scores, calib)
    return {"auroc": auroc(np.array(scores), labels), "auroc_near": auroc(np.array(near), labels), "calib": calib,
            "cal_p_pos": float(cs[labels].mean()) if labels.any() else None,
            "cal_p_neg": float(cs[~labels].mean()) if (~labels).any() else None,
            "n_pos": int(labels.sum()), "n": len(labels),
            "p_actual_pos": float(np.mean(p_act)) if p_act else None, "p_stop_pos": float(np.mean(p_stop)) if p_stop else None,
            "p_actual_neg": float(np.mean(np.array(scores)[~labels])) if len(labels) else None}


def train(data: Path, out: Path, steps: int):
    eps = load(data)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"ch": CH, "cont_scale": CONT_SCALE, "term_frac": TERM_FRAC,
                                                 "cams": CAMS, "bev_margin": BEV_MARGIN, "img_scale": IMG_SCALE, "chan_off": CHAN_OFF}))
    print(f"학습 {len(eps['train'])}편 (충돌 {int(sum(e['term'][-1] for e in eps['train']))}) · "
          f"검증 {len(eps['val'])}편 (충돌 {int(sum(e['term'][-1] for e in eps['val']))})", flush=True)
    wm = build()
    rng, t0, log = np.random.default_rng(0), time.time(), []
    for step in range(1, steps + 1):
        _, _, m = wm._train(batch(eps["train"], rng))
        if step % 200 == 0:
            print(f"step {step}: 화면 {float(m['image_loss']):.1f} · 충돌 {float(m['cont_loss']):.4f} · KL {float(m['kl']):.2f}"
                  f" · {(time.time() - t0) / 60:.0f}분", flush=True)
        if step % 2000 == 0 or step == steps:
            wm.eval()
            r = evaluate(wm, eps["val"][::max(1, len(eps["val"]) // VAL_MAX)][:VAL_MAX])   # 학습 중 검증은 고르게 VAL_MAX편 (시간 절약)
            wm.train()
            torch.save(wm.state_dict(), out / f"wm_{step}.pt")
            log.append({"step": step, **r})
            (out / "train_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False))
            print(f"  검증 {r} → wm_{step}.pt", flush=True)


def best(out: Path) -> Path:
    log = json.loads((out / "train_log.json").read_text())
    return out / f"wm_{max(log, key=lambda r: r['auroc'])['step']}.pt"


@torch.no_grad()
def show(wm, e, t, png: Path):
    """시점 t에서 세 줄 — 실제 다음 H칸 BEV / 실제 행동으로 상상 / 멈춤으로 상상. 줄마다 H칸 안 충돌 확률."""
    from PIL import Image, ImageDraw
    L = len(e["bev"])
    rows = []
    real = [e["bev"][k] if k < L else np.zeros((RES, RES, 3), np.uint8) for k in range(t + 1, t + 1 + H)]
    rows.append(("real future", real, 1.0 if e["term"][-1] and L - 1 - t <= H else 0.0))
    for name, fut in (("imagined: same action", None), ("imagined: stop", np.zeros((L, H, 2), np.float32))):
        p, prior = episode_risk(wm, e, fut)
        feat = wm.dynamics.get_feat({k: v[t:t + 1] for k, v in prior.items()})
        img = (wm.heads["decoder"](feat)["image"].mode()[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()   # BEV 부분
        rows.append((name, list(img), float(p_hit(p[t]))))
    z = 3
    W, Hh = H * RES * z + 160, len(rows) * (RES * z + 8)
    canvas = Image.new("RGB", (W, Hh), "white")
    dr = ImageDraw.Draw(canvas)
    for r, (name, frames, ph) in enumerate(rows):
        y0 = r * (RES * z + 8)
        dr.text((4, y0 + 4), f"{name}\nP(hit in 2s) {ph:.2f}", fill="black")      # 기본 글꼴은 한글이 안 나온다
        for k, f in enumerate(frames):
            canvas.paste(Image.fromarray(np.kron(f, np.ones((z, z, 1), np.uint8))), (160 + k * RES * z, y0))
    canvas.save(png)


class Lookahead:
    """주행 중 앞보기 (C): 5Hz로 BEV·직전 구간 행동을 넣어 상태를 갱신하고, 후보 (v, w)마다 2초를 samples번 상상해
    (충돌 위험 = H칸 안 충돌 확률, 충돌까지 기대 시간 s)를 낸다. isaac/nav.choose_speed가 이걸로 속도를 고른다."""

    def __init__(self, ckpt: Path, samples=8):
        cfg = ckpt.parent / "config.json"                         # 학습 때 입력 (없으면 v2 방식: BEV만·앞 카메라·지도로 거른 초록)
        cfg = json.loads(cfg.read_text()) if cfg.exists() else {}
        self.ch, self.cams, self.bev_margin = cfg.get("ch", 3), cfg.get("cams", 1), cfg.get("bev_margin", 0.15)
        self.chan_off = cfg.get("chan_off", "")
        self.wm = build(self.ch)
        self.wm.load_state_dict(torch.load(ckpt, map_location=DEV))
        self.wm.eval()
        self.samples, self.state = samples, None
        log = ckpt.parent / "train_log.json"                     # 같은 단계의 검증 보정값이 있으면 쓴다
        rows = json.loads(log.read_text()) if log.exists() else []
        self.calib = next((r.get("calib") for r in rows if f"wm_{r['step']}.pt" == ckpt.name), None)

    def reset(self):
        self.state = None

    @torch.no_grad()
    def observe(self, bev, act, rgb=None):
        img = np.concatenate([bev, rgb], -1) if self.ch == 6 else chan_off(np.asarray(bev), self.chan_off)
        data = self.wm.preprocess({"image": img[None, None],
                                   "action": (np.asarray(act, np.float32) / ACT_SCALE)[None, None],
                                   "is_first": np.array([[float(self.state is None)]], np.float32),
                                   "is_terminal": np.zeros((1, 1), np.float32)})
        embed = self.wm.encoder(data)
        self.state, _ = self.wm.dynamics.obs_step(self.state, data["action"][:, 0], embed[:, 0], data["is_first"][:, 0])

    @torch.no_grad()
    def risks(self, cands):
        K, S = len(cands), self.samples
        init = {k: v.repeat_interleave(K * S, 0) for k, v in self.state.items()}
        c = np.asarray(cands, np.float32)                         # (K, 2) 후보마다 고정 행동, 또는 (K, H, 2) 후보마다 2초 행동열
        fut = np.repeat((c if c.ndim == 3 else np.repeat(c[:, None], H, 1)) / ACT_SCALE, S, 0)
        prior = self.wm.dynamics.imagine_with_action(torch.tensor(fut, device=DEV), init)
        p = (1 - self.wm.heads["cont"](self.wm.dynamics.get_feat(prior)).mean.squeeze(-1)).cpu().numpy().reshape(K, S, H)
        surv = np.cumprod(1 - p, axis=-1)
        first = p * np.concatenate([np.ones((K, S, 1)), surv[..., :-1]], -1)       # k칸에서 처음 충돌할 확률
        t_hit = (first * np.arange(1, H + 1) * 0.2).sum(-1) + surv[..., -1] * (H + 1) * 0.2
        risk = calibrated((1 - surv[..., -1]).mean(1), self.calib)
        return list(zip(np.atleast_1d(risk).tolist(), t_hit.mean(1).tolist()))


def selftest():
    """데이터 없이: 가짜 에피소드 3편으로 배치·상상·평가가 모양대로 도는지, AUROC 계산이 맞는지."""
    assert auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75
    cal = platt(np.r_[np.full(90, 0.5), np.full(10, 0.9)], np.r_[np.zeros(85), np.ones(5), np.zeros(5), np.ones(5)])
    cp = calibrated([0.5, 0.9], cal)
    assert abs(cp[0] - 5 / 90) < 0.01 and abs(cp[1] - 0.5) < 0.01, cp   # 보정 뒤 확률 = 실제 비율
    rng = np.random.default_rng(0)
    fake = [{"ep": k, "bev": rng.integers(0, 2, (30, RES, RES, 3)).astype(np.uint8) * 255,
             "img": rng.integers(0, 2, (30, RES, RES, CH)).astype(np.uint8) * 255,
             "act": rng.uniform(-1, 1, (30, 2)).astype(np.float32), "term": np.r_[np.zeros(29), k % 2].astype(np.float32)}
            for k in range(3)]
    wm = build()
    _, _, m = wm._train(batch(fake, rng))
    wm.eval()
    p, _ = episode_risk(wm, fake[1])
    assert p.shape == (30, H) and ((p >= 0) & (p <= 1)).all()
    r = evaluate(wm, fake)
    assert r["n_pos"] > 0 and 0 <= r["auroc"] <= 1, r
    out = Path("/tmp/wm_selftest")
    out.mkdir(exist_ok=True)
    torch.save(wm.state_dict(), out / "wm.pt")
    (out / "config.json").write_text(json.dumps({"ch": CH}))
    la = Lookahead(out / "wm.pt", samples=4)
    for k in range(3):
        la.observe(fake[0]["bev"][k], fake[0]["act"][k], np.zeros((RES, RES, 3), np.uint8))
    rk = la.risks([(1.0, 0.0), (0.0, 0.0), (-0.3, 0.0)])
    show(wm, fake[1], 10, out / "show.png")
    assert len(rk) == 3 and all(0 <= p <= 1 and 0 < t <= (H + 1) * 0.2 + 1e-6 for p, t in rk), rk
    print("selftest OK —", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--cont-scale", type=float, default=CONT_SCALE)
    ap.add_argument("--term-frac", type=float, default=TERM_FRAC)
    ap.add_argument("--rgb", action="store_true", help="RGB-D: BEV(깊이)에 앞 카메라 RGB를 더한 6채널 입력")
    ap.add_argument("--img-scale", type=float, default=1.0, help="화면 복원 손실 가중치 (기본 1)")
    ap.add_argument("--chan-off", default="", help="끌 BEV 채널: R(지도)·G(장애물)·B(바닥) 중 (예 R, RB)")
    ap.add_argument("--val-max", type=int, default=10 ** 9, help="학습 중 2,000스텝마다 검증할 에피소드 수 상한 (긴 v5 에피소드 = 100)")
    ap.add_argument("--cams", type=int, default=1, help="데이터의 깊이 카메라 수 (v5 = 3, 360°) — config.json에 적어 주행 때 같은 BEV를 만든다")
    ap.add_argument("--bev-margin", type=float, default=0.15, help="BEV 초록을 지도 몇 m 안에서 지웠나 (v5 = 0, 안 지움)")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", type=int, nargs="*", help="검증 충돌 에피소드 몇 편을 그림으로 (충돌 1.4초 전 시점)")
    a = ap.parse_args()
    CONT_SCALE, TERM_FRAC = a.cont_scale, a.term_frac
    CAMS, BEV_MARGIN, VAL_MAX, IMG_SCALE, CHAN_OFF = a.cams, a.bev_margin, a.val_max, a.img_scale, a.chan_off
    if a.eval and (a.out.expanduser() / "config.json").exists():
        CHAN_OFF = json.loads((a.out.expanduser() / "config.json").read_text()).get("chan_off", "")
    CH = 6 if a.rgb else 3
    if a.eval and (a.out.expanduser() / "config.json").exists():              # 평가는 학습 때 채널을 따른다
        CH = json.loads((a.out.expanduser() / "config.json").read_text())["ch"]
    if a.selftest:
        selftest()
    if a.train:
        train(a.data.expanduser(), a.out.expanduser(), a.steps)
    if a.eval:
        wm = build()
        wm.load_state_dict(torch.load(best(a.out.expanduser()), map_location=DEV))
        wm.eval()
        val = load(a.data.expanduser())["val"]
        r = evaluate(wm, val)
        (a.out.expanduser() / "eval.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        log_f = a.out.expanduser() / "train_log.json"            # 앞보기(Lookahead)가 이 보정값을 읽는다
        log = json.loads(log_f.read_text())
        for row in log:
            if f"wm_{row['step']}.pt" == best(a.out.expanduser()).name:
                row.update(r)
        log_f.write_text(json.dumps(log, indent=1, ensure_ascii=False))
        print(r, flush=True)
        hits = [e for e in val if e["term"][-1] and len(e["bev"]) > 12]
        for e in (hits[:a.show[0]] if a.show else []):
            show(wm, e, len(e["bev"]) - 8, a.out.expanduser() / f"show_ep{e['ep']:05d}.png")
