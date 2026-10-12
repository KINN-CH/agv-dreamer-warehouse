#!/usr/bin/env python3
"""10/12 발표용 애니메이션 (10-12 사용자 요청): 깊이 카메라 → 2D 지도(BEV) → Dreamer 입력 → 상상·속도 결정을 이해하기 쉽게.
실제 데이터: FC '사람 가로지름' 0.2초 칸 140 (7장 그림과 같은 순간) — 깊이 원본·2D 지도·행동 = results/v5c/demo/fc_crossing_vis.npz,
Dreamer 상상·위험 = 같은 칸까지 observe 후 속도 5가지 2초 상상 (scratchpad anim/wm_imagine.py, Orin 컨테이너, 곧은 일정 속도 후보).
깊이 → 점: 감지 카메라 160×120, 초점 0.6 (가로 화각 120°), 차체 앞 0.36m — 이 계산으로 찍은 칸이 실제 지도와 장애물 9/9 · 바닥 94% 일치.

    python3 docs/make_anim_1012.py <wm_140.npz> <out.mp4> [--test]
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/ch/agv-wm")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"          # 비교 영상들과 같은 글꼴 (받은 지마켓 웹폰트엔 띄어쓰기 글자가 없음)
W, H, FPS = 1920, 1080, 30
BG = np.array([247, 248, 250], np.uint8)
INK, GRAY, LIGHT, BLUE, RED, GREEN = (38, 38, 38), (107, 114, 128), (203, 208, 216), (0, 84, 255), (220, 38, 38), (22, 163, 74)
FLOOR_C, OBS_C, DROP_C = np.array([30, 70, 255]), np.array([0, 200, 70]), np.array([170, 175, 185])
K = 140


def font(s):
    return ImageFont.truetype(FONT, s, index=1)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ph(t, a, b):                                                       # 구간 [a, b]에서 0→1 (부드럽게)
    return ease((t - a) / (b - a))


# ── 데이터 ─────────────────────────────────────────────────────────
z = np.load(ROOT / "results/v5c/demo/fc_crossing_vis.npz")
WM = np.load(sys.argv[1])
D = z["depth"][K].astype(np.float32)
hh, ww = D.shape
FX = 0.6 * 10 / 20.955 * ww                                            # 감지 카메라 초점 (가로 조리개 20.955)
vv, uu = np.mgrid[0:hh, 0:ww] + 0.5
Xc, Yc = (uu - ww / 2) * D / FX, (vv - hh / 2) * D / FX
fwd, left = 0.36 + D, -Xc
HC = float(np.median(Yc[-15:, :]))                                     # 카메라 높이 (바닥 줄로 맞춤 ≈ 0.33m)
up = HC - Yc
near = np.hypot(fwd, left) < 5.0
cls = np.where(near & (up > 0.1) & (up < 1.0), 2, np.where(near & (up < 0.05), 1, 0))   # 2 장애물 · 1 바닥 · 0 버림
dep_col = (cm.get_cmap("turbo")(1 - np.clip(D / 10.0, 0, 1))[..., :3] * 255).astype(np.uint8)
P = np.stack([fwd, left, up], -1).reshape(-1, 3)
C0 = dep_col.reshape(-1, 3).astype(np.float32)
CL = cls.reshape(-1)
UV = np.stack([uu, vv], -1).reshape(-1, 2)
VIS3 = (P[:, 0] < 14) & (np.abs(P[:, 1]) < 10) & (P[:, 2] < 3.5)       # 3D 화면에 보일 점
person = (CL == 2) & (P[:, 0] < 3.5) & (np.abs(P[:, 1]) < 2.5)
PC = P[person].mean(0) if person.any() else np.array([2.0, 0.0, 0.8])
person_d = float(np.median(D.reshape(-1)[person])) if person.any() else 1.8
BEV = z["bev"].copy()
BEV[..., 0] = 0                                                        # 모델 입력과 같게 (지도 채널 끔)
# 앞 카메라 점만으로 찍은 칸 (실제 지도의 앞쪽과 같음)
r_ = np.floor(44 - P[:, 0] / 0.15).astype(int)
q_ = np.floor(32 - P[:, 1] / 0.15).astype(int)
inb = (r_ >= 0) & (r_ < 64) & (q_ >= 0) & (q_ < 64) & (CL > 0)
FRONT = np.zeros((64, 64), np.int8)
for c in (1, 2):                                                       # 바닥 먼저, 장애물이 덮음 (nav.bev 순서)
    m = inb & (CL == c)
    FRONT[r_[m], q_[m]] = c
RISK, IMAG, SPEEDS = WM["risk"], WM["imag"], WM["speeds"]
PICK = max(i for i, s in enumerate(SPEEDS) if RISK[i] < 0.05 and s == max(SPEEDS[RISK < 0.05]))

# ── 3D 투영 ────────────────────────────────────────────────────────
TOP_C, TOP_T, TOP_F = np.array([1.8, 0.0, 12.0]), np.array([1.8, 0.0, 0.0]), 1040.0   # 위에서: 0.15m = 13px, BEV 832px
CX, CY = 960, 560


def camera(t):
    """시점: 비스듬히 뒤에서 (천천히 돎) → 4장 끝에 바로 위에서."""
    az = math.radians(-28 + 34 * ph(t, 3.5, 15.0))
    T0 = np.array([3.0, -0.2, 0.25])
    r0, h0 = 8.2, 3.6
    C_obl = T0 + np.array([-r0 * math.cos(az), r0 * math.sin(az), h0])
    e = ph(t, 15.0, 17.8)
    C = (1 - e) * C_obl + e * TOP_C
    T = (1 - e) * T0 + e * TOP_T
    U = (1 - e) * np.array([0, 0, 1.0]) + e * np.array([1.0, 0, 0])
    F = (1 - e) * 1250 + e * TOP_F
    f = (T - C) / np.linalg.norm(T - C)
    r = np.cross(f, U / np.linalg.norm(U))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return C, f, r, u, F


def project(X, cam):
    C, f, r, u, F = cam
    d = X - C
    zc = d @ f
    zc = np.maximum(zc, 1e-3)
    return CX + F * (d @ r) / zc, CY - F * (d @ u) / zc, zc


def splat(img, sx, sy, col, size):
    xi, yi = np.round(sx).astype(int), np.round(sy).astype(int)
    for dx in range(size):
        for dy in range(size):
            x, y = xi + dx - size // 2, yi + dy - size // 2
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            img[y[ok], x[ok]] = col[ok]


def mix(col, a):
    return (col * a[:, None] + BG * (1 - a[:, None])).astype(np.uint8)


# ── 그리기 도우미 ───────────────────────────────────────────────────
STEPS = ["깊이", "3D 점", "높이로 나누기", "위에서 보기", "칸에 찍기", "Dreamer 입력", "상상 · 속도 결정"]
CAPS = ["깊이 카메라 — 화면의 점마다 '얼마나 먼지'(m)를 잰다",
        "거리 + 카메라 위치·방향 → 점마다 실제 3D 위치 (점 1.9만 개)",
        "높이로 나누기 — 바닥 = 지나갈 수 있는 곳 · 바닥 위 0.1~1m = 장애물 · 나머지는 버림",
        "높이를 지우고 위에서 내려다본다",
        "0.15m 칸에 찍으면 64×64 칸 2D 지도 — 옆·뒤 카메라 3대도 같은 방법",
        "0.2초마다 새 지도 + 그때의 행동(속도·회전)을 Dreamer에 → 기억 갱신",
        "지도를 보지 않고 속도 5가지로 2초 뒤를 상상 → 부딪힐 확률 5% 미만 중 가장 빠르게"]
T_STEP = [0.0, 3.5, 9.5, 15.0, 19.0, 25.5, 31.5]
T_END = 42.0


def chrome(im, t):
    d = ImageDraw.Draw(im)
    k = max(i for i, a in enumerate(T_STEP) if t >= a)
    a = ph(t, T_STEP[k], T_STEP[k] + 0.5)
    d.ellipse([40, 30, 100, 90], fill=BLUE)
    d.text((70, 60), str(k + 1), font=font(36), fill="white", anchor="mm")
    col = tuple(int(c * a + b * (1 - a)) for c, b in zip(INK, BG))
    fs = 38
    while d.textlength(CAPS[k], font=font(fs)) > 1760 and fs > 26:
        fs -= 2
    d.text((122, 60), CAPS[k], font=font(fs), fill=col, anchor="lm")
    x, y0 = 60, 1042                                                   # 아래 진행 막대
    for i, s in enumerate(STEPS):
        c = BLUE if i == k else (INK if i < k else LIGHT)
        d.ellipse([x, y0 - 9, x + 18, y0 + 9], fill=c)
        d.text((x + 28, y0), s, font=font(24), fill=c, anchor="lm")
        x += 28 + d.textlength(s, font=font(24)) + 44
    return d


def bev_img(b, s, agv=True, outline="white"):
    im = Image.fromarray(np.kron(b, np.ones((s, s, 1), np.uint8)))
    if agv:
        d = ImageDraw.Draw(im)
        c, r0 = 32 * s, 44 * s
        d.rectangle([c - 0.33 / 0.15 * s, r0 - 0.40 / 0.15 * s, c + 0.33 / 0.15 * s, r0 + 1.03 / 0.15 * s], outline=outline, width=max(1, s // 3))
    return im


def arrow(d, a, b, col=BLUE, w=6):
    d.line([a, b], fill=col, width=w)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    L = 18
    d.polygon([b, (b[0] - L * math.cos(ang - 0.45), b[1] - L * math.sin(ang - 0.45)),
               (b[0] - L * math.cos(ang + 0.45), b[1] - L * math.sin(ang + 0.45))], fill=col)


def agv_box(d, cam, alpha=1.0):
    xs, ys, zs = (-1.03, 0.40), (-0.33, 0.33), (0.0, 0.30)
    V = np.array([(x, y, z_) for x in xs for y in ys for z_ in zs])
    faces = [((0, 2, 6, 4), (-1, 0, 0)), ((1, 3, 7, 5), (-1, 0, 0)), ((0, 1, 5, 4), (0, -1, 0)), ((2, 3, 7, 6), (0, 1, 0)),
             ((0, 1, 3, 2), (-1, 0, 0)), ((4, 5, 7, 6), (1, 0, 0))]
    faces = [((1, 3, 7, 5), (0, 0, 1)), ((4, 5, 7, 6), (1, 0, 0)), ((0, 1, 3, 2), (-1, 0, 0)), ((2, 3, 7, 6), (0, 1, 0)),
             ((0, 1, 5, 4), (0, -1, 0))]
    sx, sy, zc = project(V, cam)
    C = cam[0]
    shade = {(0, 0, 1): (245, 158, 11), (1, 0, 0): (217, 119, 6), (-1, 0, 0): (217, 119, 6), (0, 1, 0): (180, 95, 6), (0, -1, 0): (180, 95, 6)}
    out = []
    for idx, n in faces:
        ctr = V[list(idx)].mean(0)
        if np.dot(np.array(n, float), C - ctr) <= 0:
            continue
        out.append((np.mean(zc[list(idx)]), idx, shade[n]))
    for _, idx, colr in sorted(out, reverse=True):
        colr = tuple(int(c * alpha + b * (1 - alpha)) for c, b in zip(colr, BG))
        d.polygon([(sx[i], sy[i]) for i in idx], fill=colr, outline=(120, 60, 0))


def floor_grid(d, cam, a=1.0):
    col = tuple(int(c * a + b * (1 - a)) for c, b in zip((215, 219, 226), BG))
    for x in range(-2, 13):
        p = np.array([[x, -8, 0], [x, 8, 0]], float)
        sx, sy, _ = project(p, cam)
        d.line([(sx[0], sy[0]), (sx[1], sy[1])], fill=col, width=1)
    for y in range(-8, 9):
        p = np.array([[-2, y, 0], [12, y, 0]], float)
        sx, sy, _ = project(p, cam)
        d.line([(sx[0], sy[0]), (sx[1], sy[1])], fill=col, width=1)


# ── 장면 ──────────────────────────────────────────────────────────
VIEW = Image.fromarray(z["view"][K][:, 320:]).resize((800, 600), Image.BICUBIC)
DEP_BIG = Image.fromarray(dep_col).resize((800, 600), Image.NEAREST)
IMG_X0, IMG_Y0, IMG_S = 1000, 230, 5.0                                 # 1장: 깊이 화면 위치 (160×120 → 800×600)


def frame(t):
    im = Image.new("RGB", (W, H), tuple(BG))
    if t < 3.5 + 0.01:                                                 # ① 깊이 카메라
        a = ph(t, 0.0, 0.6)
        im.paste(VIEW, (120, 230))
        im.paste(DEP_BIG, (IMG_X0, IMG_Y0))
        d = chrome(im, t)
        d.text((120, 200), "실제 장면 (참고용 3인칭)", font=font(28), fill=INK, anchor="lb")
        d.text((IMG_X0, 200), "AGV 앞 깊이 카메라 원래 화면", font=font(28), fill=INK, anchor="lb")
        pu, pv = UV[person].mean(0) if person.any() else (80, 60)
        px, py = IMG_X0 + pu * IMG_S, IMG_Y0 + pv * IMG_S
        if t > 1.2:
            d.ellipse([px - 46, py - 46, px + 46, py + 46], outline="white", width=5)
            d.rounded_rectangle([px + 56, py - 34, px + 330, py + 34], 12, fill="white")
            d.text((px + 193, py), f"사람: 약 {person_d:.1f}m", font=font(30), fill=INK, anchor="mm")
        gx, gy = IMG_X0, 860                                           # 색 막대
        for i in range(400):
            c = (np.array(cm.get_cmap("turbo")(1 - i / 400)[:3]) * 255).astype(int)
            d.line([(gx + 200 + i, gy), (gx + 200 + i, gy + 22)], fill=tuple(c))
        d.text((gx + 190, gy + 11), "가까움", font=font(24), fill=INK, anchor="rm")
        d.text((gx + 612, gy + 11), "멂 (10m)", font=font(24), fill=INK, anchor="lm")
        if a < 1:
            im = Image.blend(Image.new("RGB", (W, H), tuple(BG)), im, a)
            chrome(im, t)
        return im
    if t < 19.0:                                                       # ②③④ 3D 점
        cam = camera(t)
        d = ImageDraw.Draw(im)
        g = ph(t, 4.5, 6.0)
        floor_grid(d, cam, g)
        hc = 1 - ph(t, 15.0, 17.8)                                     # ④ 높이를 지운다
        X = P.copy()
        X[:, 2] *= hc
        sx, sy, zc = project(X, cam)
        m = ph(t, 3.5, 6.2)                                            # ② 화면 → 3D 위치로 날아감
        ix, iy = IMG_X0 + UV[:, 0] * IMG_S, IMG_Y0 + UV[:, 1] * IMG_S
        sx, sy = (1 - m) * ix + m * sx, (1 - m) * iy + m * sy
        rc = ph(t, 10.3, 11.8)                                         # ③ 색을 나눔
        target = np.where(CL[:, None] == 2, OBS_C, np.where(CL[:, None] == 1, FLOOR_C, DROP_C))
        col = (1 - rc) * C0 + rc * target
        a = np.ones(len(P))
        a[~VIS3] = 1 - ph(t, 4.2, 5.6)                                 # 화면 밖으로 갈 먼 점
        a[CL == 0] *= 1 - ph(t, 11.8, 13.0)                            # 버리는 점
        keep = a > 0.04
        order = np.argsort(-zc[keep])
        size = int(round(5 - 2.6 * m))
        if t > 17.8:
            size = 3
        if 4.5 < t:
            agv_box(d, cam, g)
        arr = np.asarray(im).copy()
        idx = np.where(keep)[0][order]
        splat(arr, sx[idx], sy[idx], mix(col[idx], a[idx]), max(size, 2))
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        if 9.5 <= t < 16.5:                                            # ③ 높이 0.1m · 1.0m 판
            pa = ph(t, 9.5, 10.3) * (1 - ph(t, 15.0, 16.0))
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            do = ImageDraw.Draw(ov)
            for zz, colr, lab in ((1.0 * hc, (220, 38, 38), "1.0m"), (0.1 * hc, (22, 163, 74), "0.1m")):
                Q = np.array([[0.4, -5, zz], [6.5, -5, zz], [6.5, 5, zz], [0.4, 5, zz]])
                qx, qy, _ = project(Q, cam)
                do.polygon(list(zip(qx, qy)), fill=colr + (int(38 * pa),), outline=colr + (int(200 * pa),))
                do.text((qx[0] - 12, qy[0]), lab, font=font(30), fill=colr + (int(255 * pa),), anchor="rm")
            im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
            d = ImageDraw.Draw(im)
        if 6.4 < t < 15.0:                                             # 사람 표시
            la = ph(t, 6.4, 7.0) * (1 - ph(t, 14.4, 15.0))
            hx, hy, _ = project(PC[None] * np.array([1, 1, hc]), cam)
            colr = tuple(int(c * la + b * (1 - la)) for c, b in zip(INK, BG))
            d.line([(hx[0] + 10, hy[0] - 30), (hx[0] + 120, hy[0] - 130)], fill=colr, width=3)
            d.text((hx[0] + 126, hy[0] - 136), "사람", font=font(32), fill=colr, anchor="lb")
        if 4.5 < t < 9.5:
            cx_, cy_, _ = project(np.array([[0.36, 0, HC]]), cam)
            la = ph(t, 5.0, 5.6) * (1 - ph(t, 9.0, 9.5))
            colr = tuple(int(c * la + b * (1 - la)) for c, b in zip(INK, BG))
            d.text((cx_[0] - 20, cy_[0] + 40), "AGV · 깊이 카메라", font=font(28), fill=colr, anchor="rt")
        if 12.0 < t < 15.0:
            la = ph(t, 12.0, 12.6) * (1 - ph(t, 14.5, 15.0))
            n_obs, n_flo, n_drop = int((CL == 2).sum()), int((CL == 1).sum()), int((CL == 0).sum())
            for k_, (txt, colr) in enumerate(((f"장애물 {n_obs}개", GREEN), (f"바닥 {n_flo:,}개", BLUE), (f"버림 {n_drop:,}개 (높거나 5m 밖)", GRAY))):
                c2 = tuple(int(c * la + b * (1 - la)) for c, b in zip(colr, BG))
                d.text((1400, 820 + k_ * 46), txt, font=font(28), fill=c2, anchor="lm")
        chrome(im, t)
        return im
    if t < 25.5 + 0.01:                                                # ⑤ 칸에 찍기
        cam = camera(t)
        d = ImageDraw.Draw(im)
        cs, x0, y0 = 13, 544, 144
        ga = ph(t, 19.0, 19.8)
        blk = ph(t, 22.2, 23.2)                                        # 지도 바탕을 검게 (본 적 없는 곳)
        bgc = tuple(int(b * (1 - blk)) for b in BG)
        d.rectangle([x0, y0, x0 + 64 * cs, y0 + 64 * cs], fill=bgc)
        lc = tuple(int(c * ga + b * (1 - ga)) for c, b in zip((205, 210, 218), BG))
        if blk < 1:
            for i in range(65):
                d.line([(x0 + i * cs, y0), (x0 + i * cs, y0 + 64 * cs)], fill=lc, width=1)
                d.line([(x0, y0 + i * cs), (x0 + 64 * cs, y0 + i * cs)], fill=lc, width=1)
        fill_t = ph(t, 20.4, 21.8)                                     # 가까운 칸부터 채움
        rr, qq = np.mgrid[0:64, 0:64]
        dist = np.hypot((44 - rr) * 0.15, (32 - qq) * 0.15)
        arr = np.asarray(im).copy()
        show = (FRONT > 0) & (dist <= fill_t * 5.2)
        other = ph(t, 22.6, 23.6)                                      # 옆·뒤 카메라
        for r in range(64):
            for q in range(64):
                ys, xs = slice(y0 + r * cs + 1, y0 + (r + 1) * cs), slice(x0 + q * cs + 1, x0 + (q + 1) * cs)
                real = BEV[K, r, q]
                if show[r, q]:
                    c = OBS_C if FRONT[r, q] == 2 else FLOOR_C
                    c = c * (1 - blk) + real * blk if real.any() else c * (1 - blk)
                    arr[ys, xs] = c.astype(np.uint8)
                elif real.any() and other > 0:
                    arr[ys, xs] = (real * other + np.array(bgc) * (1 - other)).astype(np.uint8)
        sn = ph(t, 19.8, 21.0)                                         # 점이 칸 가운데로 모였다가 사라짐
        pa = 1 - ph(t, 20.6, 21.8)
        if pa > 0.02:
            m = CL > 0
            cr = (np.floor(44 - P[m, 0] / 0.15) + 0.5)
            cq = (np.floor(32 - P[m, 1] / 0.15) + 0.5)
            sx0, sy0, _ = project(P[m] * np.array([1, 1, 0]), cam)
            sx1, sy1 = x0 + cq * cs, y0 + cr * cs
            sx, sy = (1 - sn) * sx0 + sn * sx1, (1 - sn) * sy0 + sn * sy1
            col = np.where(CL[m][:, None] == 2, OBS_C, FLOOR_C).astype(np.float32)
            splat(arr, sx, sy, mix(col, np.full(len(sx), pa)), 3)
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        c, r0 = x0 + 32 * cs, y0 + 44 * cs                             # 차체
        oc = (217, 119, 6) if blk < 0.5 else (255, 255, 255)
        d.rectangle([c - 0.33 / 0.15 * cs, r0 - 0.40 / 0.15 * cs, c + 0.33 / 0.15 * cs, r0 + 1.03 / 0.15 * cs], outline=oc, width=3)
        la = ph(t, 23.6, 24.2)
        if la > 0:
            colr = tuple(int(cc * la + b * (1 - la)) for cc, b in zip(INK, BG))
            d.text((x0 + 64 * cs + 40, y0 + 40), "64 × 64 칸", font=font(36), fill=colr, anchor="lm")
            d.text((x0 + 64 * cs + 40, y0 + 90), "= 9.6m × 9.6m", font=font(30), fill=colr, anchor="lm")
            d.text((x0 + 64 * cs + 40, y0 + 150), "앞 6.6m · 뒤 3m · 좌우 4.8m", font=font(28), fill=colr, anchor="lm")
            for k_, (txt, cc) in enumerate((("초록 = 장애물", (0, 160, 60)), ("파랑 = 바닥(빈 곳)", (0, 0, 220)), ("검정 = 안 보이는 곳", INK))):
                c2 = tuple(int(v * la + b * (1 - la)) for v, b in zip(cc, BG))
                d.text((x0 + 64 * cs + 40, y0 + 250 + k_ * 50), txt, font=font(28), fill=c2, anchor="lm")
            d.text((x0 - 40, y0 + 40), "이것이 Dreamer가", font=font(32), fill=colr, anchor="rm")
            d.text((x0 - 40, y0 + 86), "보는 화면", font=font(32), fill=colr, anchor="rm")
        if 22.4 < t < 24.0:
            la2 = ph(t, 22.4, 22.9) * (1 - ph(t, 23.5, 24.0))
            colr = tuple(int(cc * la2 + b * (1 - la2)) for cc, b in zip(BLUE, BG))
            d.text((x0 - 40, y0 + 64 * cs - 60), "+ 옆·뒤 카메라 3대", font=font(32), fill=colr, anchor="rm")
        chrome(im, t)
        return im
    # ⑥⑦ Dreamer
    d = ImageDraw.Draw(im)
    mv = ph(t, 25.5, 26.5)                                             # 큰 지도 → 왼쪽 작은 지도
    s_big, s_small = 832, 320
    sz = int(s_big * (1 - mv) + s_small * mv)
    bx, by = int(544 * (1 - mv) + 70 * mv), int(144 * (1 - mv) + 370 * mv)
    feed = [K - 3, K - 2, K - 1, K]
    n_in = 1 if t < 27.5 else min(4, 1 + int((t - 27.5) / 0.8))        # 0.2초마다 들어오는 지도 (보기 좋게 느리게)
    if t >= 27.5:
        for j in range(4 - n_in, 0, -1):                               # 뒤에 쌓인 지난 지도
            pass
        for j, kk in enumerate(feed[:n_in]):
            off = (n_in - 1 - j) * 16
            im.paste(bev_img(BEV[kk], 5).resize((s_small, s_small), Image.NEAREST), (bx - off, by - off))
        cur = feed[n_in - 1]
    else:
        im.paste(bev_img(BEV[K], 13).resize((sz, sz), Image.NEAREST), (bx, by))
        cur = K
    d = ImageDraw.Draw(im)
    da = ph(t, 26.4, 27.2)
    if da > 0:
        def C_(c, a=da):
            return tuple(int(v * a + b * (1 - a)) for v, b in zip(c, BG))
        d.text((40, 300), "2D 지도 (0.2초마다 새로)", font=font(30), fill=C_(INK), anchor="lb")
        v_, w_ = z["act"][cur]
        d.text((70, 712), "+ 그때의 행동", font=font(28), fill=C_(INK), anchor="lt")
        d.text((70, 752), f"속도 {v_:.2f} m/s · 회전 {w_:.2f}", font=font(26), fill=C_(GRAY), anchor="lt")
        dt = (cur - K) * 0.2
        lab_ = "지금" if cur == K else f"{dt:.1f}초"
        d.rounded_rectangle([bx + s_small - 112, by + 10, bx + s_small - 10, by + 52], 10, fill=C_(BLUE))
        d.text((bx + s_small - 61, by + 31), lab_, font=font(24), fill="white", anchor="mm")
        arrow(d, (410, 530), (500, 530), C_(BLUE))
        d.rounded_rectangle([510, 250, 930, 860], 24, fill=C_((219, 234, 254)), outline=C_(BLUE), width=4)
        d.text((720, 290), "Dreamer", font=font(40), fill=C_(BLUE), anchor="mm")
        d.polygon([(545, 345), (895, 345), (850, 455), (590, 455)], fill=C_((255, 255, 255)), outline=C_(BLUE))
        d.text((720, 385), "인코더", font=font(32), fill=C_(INK), anchor="mm")
        d.text((720, 425), "지도 → 작은 숫자 묶음", font=font(24), fill=C_(GRAY), anchor="mm")
        pulse = 0.0
        if t >= 27.5:
            ph_ = ((t - 27.5) % 0.8) / 0.8
            pulse = max(0.0, 1 - abs(ph_ - 0.35) / 0.35) if n_in <= 4 and t < 27.5 + 3.2 else 0.0
        rng = np.random.default_rng(cur)
        code = rng.random((8, 8)) > 0.62
        for i in range(8):
            for j in range(8):
                cc = C_(BLUE) if code[i, j] else C_((255, 255, 255))
                d.rectangle([675 + j * 11, 480 + i * 11, 684 + j * 11, 489 + i * 11], fill=cc, outline=C_(LIGHT))
        d.text((780, 524), "z", font=font(30), fill=C_(INK), anchor="lm")
        arrow(d, (720, 575), (720, 640), C_(BLUE), 5)
        mem = tuple(int(255 * (1 - pulse) + v * pulse) for v in (191, 219, 254))
        d.rounded_rectangle([555, 650, 885, 800], 18, fill=C_(mem), outline=C_(BLUE), width=3)
        d.text((720, 700), "기억 h", font=font(34), fill=C_(INK), anchor="mm")
        d.text((720, 750), "지난 장면 + 행동을 쌓음", font=font(24), fill=C_(GRAY), anchor="mm")
        d.arc([860, 670, 920, 780], 270, 90, fill=C_(BLUE), width=4)       # 되돌이 화살표
        d.polygon([(872, 780), (888, 772), (888, 788)], fill=C_(BLUE))
        if pulse > 0:
            arrow(d, (410, 530), (500, 530), BLUE, 10)
    ia = ph(t, 31.5, 32.2)                                             # ⑦ 상상
    if ia > 0:
        def C_(c, a=ia):
            return tuple(int(v * a + b * (1 - a)) for v, b in zip(c, BG))
        arrow(d, (930, 555), (1000, 555), C_(BLUE))
        d.text((965, 520), "상상", font=font(26), fill=C_(BLUE), anchor="mb")
        d.text((965, 600), "(지도 없이)", font=font(20), fill=C_(GRAY), anchor="mt")
        hx = [1180, 1320, 1460]
        for q, st in enumerate((2, 5, 9)):
            d.text((hx[q] + 62, 200), f"{(st + 1) * 0.2:.1f}초 뒤", font=font(24), fill=C_(GRAY), anchor="mb")
        d.text((1700, 200), "부딪힐 확률", font=font(24), fill=C_(GRAY), anchor="mb")
        labels = ["1.0 m/s", "0.6 m/s", "0.3 m/s", "멈춤", "뒤로 0.3"]
        for i in range(5):
            ra = ph(t, 32.0 + i * 0.7, 32.5 + i * 0.7)
            if ra <= 0:
                continue
            y = 215 + i * 150
            d.text((1150, y + 64), labels[i], font=font(32), fill=C_(INK, ra), anchor="rm")
            for q, st in enumerate((2, 5, 9)):
                qa = ph(t, 32.1 + i * 0.7 + q * 0.12, 32.4 + i * 0.7 + q * 0.12)
                if qa > 0:
                    tile = bev_img(IMAG[i, st], 2)
                    tile = Image.blend(Image.new("RGB", tile.size, tuple(BG)), tile, qa)
                    im.paste(tile, (hx[q], y))
            d = ImageDraw.Draw(im)
            rk = float(RISK[i])
            ba = ph(t, 32.4 + i * 0.7, 32.9 + i * 0.7)
            colr = RED if rk >= 0.05 else GREEN
            bw = max(4, int(min(1.0, rk / 0.25) * 180 * ba))
            d.rectangle([1610, y + 44, 1610 + bw, y + 84], fill=C_(colr, ba))
            txt = f"{rk * 100:.0f}%" if rk >= 0.1 else f"{rk * 100:.1f}%"
            d.text((1610 + bw + 14, y + 64), txt, font=font(32), fill=C_(colr, ba), anchor="lm")
        ha = ph(t, 36.0, 36.6)
        if ha > 0:
            y = 215 + PICK * 150
            d.rounded_rectangle([1010, y - 12, 1890, y + 140], 18, outline=C_(BLUE, ha), width=6)
            d.rounded_rectangle([1010, y - 52, 1150, y - 10], 12, fill=C_(BLUE, ha))
            d.text((1080, y - 31), "선택", font=font(28), fill="white", anchor="mm")
            msg = f"이 순간 실제 출력: 그대로(1.0m/s) 가면 {RISK[0] * 100:.0f}% → {SPEEDS[PICK]:.1f}m/s로 낮춰 지나감"
            fs = 30
            while d.textlength(msg, font=font(fs)) > 880 and fs > 20:
                fs -= 1
            d.text((1010, 992), msg, font=font(fs), fill=C_(INK, ha), anchor="lb")
    chrome(im, t)
    return im


def main(out, test=False):
    if test:
        for t in (2.5, 4.8, 7.5, 10.8, 12.5, 18.5, 21.0, 24.5, 29.0, 37.5):
            frame(t).save(str(out).replace(".mp4", f"_t{t:04.1f}.png"))
        print("시험 장면 저장")
        return
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    n = int(T_END * FPS)
    for k in range(n):
        p.stdin.write(np.asarray(frame(k / FPS)).tobytes())
    p.stdin.close()
    p.wait()
    print(out, n, "장", f"{T_END}초")


if __name__ == "__main__":
    main(Path(sys.argv[2]), "--test" in sys.argv)
