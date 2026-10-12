#!/usr/bin/env python3
"""10/12 발표용 긴 영상 (10-11 사용자 요청): 출발부터 도착까지 한 주행을 한 화면에 —
왼쪽 = AGV를 따라 위에서 본 화면 + 바닥에 지금 A* 경로(파랑)·확정 장애물 칸(주황) · 오른쪽 = 앞 깊이 카메라 / 2D 지도(Dreamer 입력) /
Dreamer가 상상한 0.6·1.2·2.0초 뒤 ('그대로 가면' · '고른 속도') + 속도 5가지 충돌 위험 / 창고 전체 길찾기 지도.
입력 = logistics.py --eval5 --dump-vis --follow-cam H --log-look 가 남긴 *_vis.pkl.gz (VM ~/agv/panel1.sh). 상상을 그리려면 월드모델·GPU (VM):
    ~/isaacsim-venv/bin/python make_panel_1012.py <vis.pkl.gz> <out.mp4> "<제목>" [축 매핑 a,b,c,d]
"""
import gzip
import math
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

AGV = Path.home() / "agv"
sys.path[:0] = [str(AGV / "src"), str(AGV / "isaac")]
FONT = str(AGV / "NotoSansCJK-Bold.ttc")
SPEEDS = (1.0, 0.6, 0.3, 0.0, -0.3)                                 # nav.SPEEDS (Isaac 없이 읽으려고 옮겨 적음)
W_, H_ = 1920, 1080
BLUE, ORANGE, RED, GREEN, INK, GRAY = (0, 84, 255), (245, 158, 11), (220, 38, 38), (22, 163, 74), (38, 38, 38), (107, 114, 128)


def font(s):
    return ImageFont.truetype(FONT, s, index=1)


def turbo(d, near=0.3, far=8.0):
    """깊이(m) → 색 (가까움 빨강 · 멂 파랑, 7장 그림과 같은 방향)."""
    x = np.clip((np.nan_to_num(d, nan=far, posinf=far) - near) / (far - near), 0, 1)
    x = 1 - x
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def imagine(D):
    """판단마다: 그 순간 Dreamer 상태에서 '그대로(1.0m/s)'와 '고른 속도' 후보 행동열로 2초 상상 → 64×64 지도 10장씩."""
    if D["cond"] != "wm" or not D.get("wmlog"):
        return {}
    import torch
    import wm_bev
    la = wm_bev.Lookahead(Path(open(AGV / "wm_bev_v5c.best").read().strip()))
    wm, out = la.wm, {}
    with torch.no_grad():
        for e in D["wmlog"]:
            ks = [0, SPEEDS.index(e["pick"]) if e["pick"] in SPEEDS else 0]
            init = {k: torch.tensor(v, device=wm_bev.DEV).repeat_interleave(2, 0) for k, v in e["state"].items()}
            fut = torch.tensor(np.asarray(e["cands"])[ks] / wm_bev.ACT_SCALE, device=wm_bev.DEV)
            prior = wm.dynamics.imagine_with_action(fut, init)
            img = wm.heads["decoder"](wm.dynamics.get_feat(prior))["image"].mode()[..., :3].clamp(0, 1)
            out[e["i"]] = (img * 255).byte().cpu().numpy()            # (2, 10, 64, 64, 3)
    return out


def main(src, dst, title, axes=(0, -1, -1, 0)):
    D = pickle.load(gzip.open(src))
    view, vt, pose = D["view"], np.asarray(D["view_t"]), np.asarray(D["pose"])
    N = len(D["bev"])
    imag = imagine(D)
    wm_at = {e["i"]: e for e in D.get("wmlog") or []}
    nav = D["navlog"]
    nb = D["navbase"]
    K, Hc = np.asarray(D["cam_K"]), D["follow_cam"]
    a, b, c, d = axes                                                # 바닥 (dX, dY) → 화면 (u, v) 방향 (시험 화면으로 맞춤)
    occ = nb["occ"]
    # 창고 전체 지도 (오른쪽 아래): 랙·벽 = 짙은 회색
    MW, MH = 820, 290
    sc = min(MW / (occ.shape[1] * nb["cell"]), MH / (occ.shape[0] * nb["cell"]))
    base_map = Image.fromarray(np.where(occ[::-1], 70, 245).astype(np.uint8)).convert("RGB").resize(
        (int(occ.shape[1] * nb["cell"] * sc), int(occ.shape[0] * nb["cell"] * sc)), Image.NEAREST)

    def to_map(x, y):                                                # 세계 → 전체 지도 픽셀 (위 = +y)
        return (x - nb["x0"]) * sc, base_map.height - (y - nb["y0"]) * sc

    def floor_px(X, Y, xc, yc):                                      # 바닥 점 → 위 카메라 화면 (카메라가 AGV 바로 위)
        dX, dY = X - xc, Y - yc
        return K[0, 2] + K[0, 0] * (a * dX + b * dY) / Hc, K[1, 2] + K[1, 1] * (c * dX + d * dY) / Hc

    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_}x{H_}", "-r", "10", "-i", "-",
                            "-pix_fmt", "yuv420p", "-crf", "20", str(dst)], stdin=subprocess.PIPE)   # 한 장씩 바로 (전부 쌓으면 수 GB)
    last_rep, rep_t, old_path, prev_path, C = None, -9.0, None, None, None
    for k in range(len(view)):
        t = float(vt[k])
        i = min(int(round(t / 0.2)), N - 1)                           # 0.2초 기록 칸
        xc, yc = np.interp(t, np.arange(N) * 0.2, pose[:, 0]), np.interp(t, np.arange(N) * 0.2, pose[:, 1])
        nl = [e for e in nav if e["i"] <= i]
        cur = nl[-1] if nl else None
        if cur is not None and cur["replans"] != last_rep:           # 경로를 다시 찾은 순간
            if last_rep is not None and cur["replans"] > last_rep:
                rep_t, old_path = t, prev_path
            last_rep = cur["replans"]
        prev_path = cur["path"] if cur is not None else None
        C = Image.new("RGB", (W_, H_), "white")
        dr = ImageDraw.Draw(C)
        # ── 왼쪽: 위 카메라 + 바닥에 경로·확정 장애물 ──
        cam = Image.fromarray(view[k]).convert("RGB")
        ov = Image.new("RGBA", cam.size, (0, 0, 0, 0))
        do = ImageDraw.Draw(ov)
        if cur is not None:
            for r_, c_ in cur["seen"]:
                X, Y = nb["x0"] + (c_ + 0.5) * nb["cell"], nb["y0"] + (r_ + 0.5) * nb["cell"]
                u, v = floor_px(X, Y, xc, yc)
                s_ = K[0, 0] * nb["cell"] / Hc / 2
                do.rectangle([u - s_, v - s_, u + s_, v + s_], fill=ORANGE + (150,))
            if old_path is not None and t - rep_t < 2.0:
                do.line([floor_px(px, py, xc, yc) for px, py in old_path], fill=(120, 120, 120, 200), width=4)
            if cur["path"] is not None:
                do.line([floor_px(px, py, xc, yc) for px, py in cur["path"]], fill=BLUE + (220,), width=6)
        gx, gy = D["goal"]
        u, v = floor_px(gx, gy, xc, yc)
        do.ellipse([u - 12, v - 12, u + 12, v + 12], outline=RED + (255,), width=5)
        cam = Image.alpha_composite(cam.convert("RGBA"), ov).convert("RGB").resize((1000, 1000), Image.BICUBIC)
        C.paste(cam, (10, 55))
        who = "Dreamer" if D["cond"] == "wm" else "규칙"
        dr.text((14, 28), f"{who} — {title}", font=font(30), fill=BLUE if D["cond"] == "wm" else RED, anchor="lm")
        sp = float(np.hypot(*np.diff(pose[max(i - 1, 0):i + 1, :2], axis=0)[0]) / 0.2) if i > 0 else 0.0
        dr.text((1010, 28), f"{t:5.1f}초 · 속도 {sp:.1f} m/s", font=font(26), fill=INK, anchor="rm")
        if t - rep_t < 2.0:
            dr.rounded_rectangle([30, 90, 380, 140], 10, fill=(255, 255, 255))
            dr.text((205, 115), "막힘 발견 → 경로 다시 찾음", font=font(24), fill=BLUE, anchor="mm")
        dr.text((14, 1078), "파랑 = 지금 A* 경로 · 주황 = 확정된 장애물 칸(기록) · 빨간 원 = 목적지", font=font(20), fill=GRAY, anchor="lb")
        # ── 오른쪽 위: 앞 깊이 카메라 · 2D 지도 ──
        x0 = 1030
        dr.text((x0, 30), "앞 깊이 카메라", font=font(22), fill=INK, anchor="lm")
        C.paste(Image.fromarray(turbo(D["depth"][i])).resize((347, 260), Image.NEAREST), (x0, 48))
        dr.text((x0 + 380, 30), "2D 지도 (Dreamer 입력)" if D["cond"] == "wm" else "2D 지도 (규칙은 안 씀)", font=font(22), fill=INK, anchor="lm")
        bev = np.array(D["bev"][i])
        bev[..., 0] = 0                                              # 모델은 지도(R) 채널을 끄고 본다 (chan_off R)
        C.paste(Image.fromarray(bev).resize((260, 260), Image.NEAREST), (x0 + 380, 48))
        # ── 오른쪽 가운데: 상상 · 속도별 충돌 위험 ──
        y0 = 330
        e = wm_at.get(i) or wm_at.get(max([j for j in wm_at if j <= i], default=-1))
        if D["cond"] == "wm" and (e is None or e["i"] not in imag):
            dr.text((x0, y0 + 2), "Dreamer의 상상 — 준비 중", font=font(22), fill=BLUE, anchor="lm")
        elif D["cond"] == "wm":
            im = imag[e["i"]]
            for r_, (lab, kk) in enumerate((("그대로 가면 (1.0 m/s)", 0), (f"고른 속도 ({e['pick']:.1f} m/s)", 1))):
                yy = y0 + 62 + r_ * 160
                risk = e["risk"][0 if kk == 0 else SPEEDS.index(e["pick"])]
                dr.text((x0, yy + 50), lab, font=font(20), fill=INK, anchor="lm")
                dr.text((x0, yy + 82), f"충돌 위험 {risk * 100:.0f}%", font=font(22), fill=RED if risk >= 0.05 else GREEN, anchor="lm")
                for q, step in enumerate((2, 5, 9)):                 # 0.6 · 1.2 · 2.0초 뒤
                    C.paste(Image.fromarray(im[kk, step]).resize((140, 140), Image.NEAREST), (x0 + 230 + q * 146, yy))
                    if r_ == 0:
                        dr.text((x0 + 300 + q * 146, yy - 14), f"{(step + 1) * 0.2:.1f}초 뒤", font=font(18), fill=GRAY, anchor="mm")
            dr.text((x0, y0 + 2), "Dreamer의 상상 — 0.2초마다 속도 5가지를 2초씩", font=font(22), fill=BLUE, anchor="lm")
            bx = x0 + 680                                            # 속도 5가지 막대
            for q, (s_, rk) in enumerate(zip(SPEEDS, e["risk"])):
                yy = y0 + 48 + q * 62
                lab = "뒤로" if s_ < 0 else f"{s_:.1f}"
                dr.text((bx, yy + 14), lab, font=font(18), fill=INK, anchor="lm")
                dr.rectangle([bx + 46, yy + 4, bx + 46 + max(2, int(120 * rk)), yy + 24], fill=RED if rk >= 0.05 else GREEN)
                dr.text((bx + 46, yy + 40), f"{rk * 100:.0f}%", font=font(16), fill=GRAY, anchor="lm")
                if s_ == e["cap"]:
                    dr.rectangle([bx - 4, yy, bx + 172, yy + 50], outline=BLUE, width=3)
        else:
            sl = (D.get("state_log") or [{}])[min(i, len(D.get("state_log") or [{}]) - 1)]
            lim = sl.get("limit")
            st = "정지 (1.5m 안에 무언가)" if lim == 0 else "감속 (3m 안에 무언가)" if lim is not None else "주행"
            dr.text((x0, y0 + 2), "규칙 — 상상하지 않고 지금 거리만 본다", font=font(22), fill=RED, anchor="lm")
            dr.text((x0, y0 + 60), "앞 1.5m 안에 무언가 있으면 정지, 3m 안이면 감속", font=font(22), fill=INK, anchor="lm")
            dr.text((x0, y0 + 120), f"지금: {st}", font=font(30), fill=RED if lim == 0 else INK, anchor="lm")
        # ── 오른쪽 아래: 창고 전체 길찾기 지도 ──
        my = 735
        dr.text((x0, my - 14), "길찾기 지도 — 처음엔 랙·벽만, 본 장애물을 기록", font=font(22), fill=INK, anchor="lm")
        M = base_map.copy()
        dm = ImageDraw.Draw(M)
        if cur is not None:
            for r_, c_ in cur["seen"]:
                px, py = to_map(nb["x0"] + (c_ + 0.5) * nb["cell"], nb["y0"] + (r_ + 0.5) * nb["cell"])
                dm.rectangle([px - 1, py - 1, px + 1, py + 1], fill=ORANGE)
            if cur["path"] is not None:
                dm.line([to_map(px, py) for px, py in cur["path"]], fill=BLUE, width=3)
        dm.line([to_map(px, py) for px, py in pose[:i + 1, :2]], fill=GRAY, width=2)
        for (px, py), col in ((D["start"], INK), (D["goal"], RED)):
            q = to_map(px, py)
            dm.ellipse([q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6], fill=col)
        q = to_map(xc, yc)
        dm.ellipse([q[0] - 7, q[1] - 7, q[0] + 7, q[1] + 7], outline=BLUE, width=3)
        C.paste(M, (x0 + (MW - M.width) // 2, my + 6))
        enc.stdin.write(np.asarray(C).tobytes())
        if k == len(view) // 2:
            C.save(str(dst).replace(".mp4", "_mid.png"))
    end = C.copy()                                                   # 끝: 결과를 크게
    de = ImageDraw.Draw(end)
    ok = D.get("arrived") and not D.get("contact")
    de.rounded_rectangle([260, 470, 760, 600], 16, fill=(255, 255, 255))
    de.text((510, 535), f"도착 {D['time_s']}초" if ok else "충돌!", font=font(56), fill=GREEN if ok else RED, anchor="mm")
    for _ in range(20):                                              # 2초 멈춤
        enc.stdin.write(np.asarray(end).tobytes())
    enc.stdin.close()
    enc.wait()
    end.save(str(dst).replace(".mp4", "_end.png"))
    print(dst, len(view) + 20, "장", D["cond"], "도착" if ok else "충돌/실패")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], tuple(map(float, a[3].split(","))) if len(a) > 3 else (0, -1, -1, 0))
