#!/usr/bin/env python3
"""10/12 중간발표 그림·영상·숫자 → docs/ppt_media_1012/. 실측 결과(results/)에서 만든다.
숫자는 numbers.json 한 곳에 — make_ppt_1012.py가 읽는다 (2회차가 끝나면 이 스크립트만 다시 돌리면 된다).

    python3 docs/make_media_1012.py && python3 docs/make_ppt_1012.py
"""
import json
import os
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

R = Path("/data/ch/agv-wm/results")
V = R / "v5c"
OLD, OLD8 = Path(__file__).parent / "ppt_media", Path(__file__).parent / "ppt_media_1008"
OUT = Path(__file__).parent / "ppt_media_1012"
OUT.mkdir(exist_ok=True)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size, index=1)


def to_mp4(frames, name, fps=5):
    w, h = frames[0].size
    w2, h2 = w // 2 * 2, h // 2 * 2
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w2}x{h2}", "-r", str(fps),
                          "-i", "-", "-pix_fmt", "yuv420p", "-crf", "18", str(OUT / f"{name}.mp4")], stdin=subprocess.PIPE)
    for f in frames + [frames[-1]] * fps * 2:
        p.stdin.write(np.asarray(f.convert("RGB").crop((0, 0, w2, h2))).tobytes())
    p.stdin.close()
    p.wait()


# ── 지난 발표에서 쓰던 실측 그림 (Cosmos · Dreamer · 돌발 종류) ────────────
for src, dst in ((OLD / "s6_real_straight.png", "cosmos_real.png"), (OLD / "s6_cosmos_straight.png", "cosmos_imag.png"),
                 (OLD / "s17_real.png", "dreamer_real.png"), (OLD / "s17_dreamer.png", "dreamer_imag.png"),
                 (OLD / "s6_real_straight.mp4", "cosmos_real.mp4"), (OLD / "s6_cosmos_straight.mp4", "cosmos_imag.mp4"),
                 (OLD8 / "s11_hazards.png", "hazards.png"), (R / "fc/views/overview.png", "fc_overview.png")):
    shutil.copy(src, OUT / dst)

# ── 창고 지도: 학습 9곳 / 시험 5곳 ─────────────────────────────────────
W_ = R / "v5c/wh_views"


def maps(names, labels, name, cols, cell=330, frame="#9CA3AF"):
    rows = math.ceil(len(names) / cols)
    canvas = Image.new("RGB", (cols * (cell + 24), rows * (cell + 60)), "white")
    d = ImageDraw.Draw(canvas)
    for k, (n, lab) in enumerate(zip(names, labels)):
        im = Image.open(W_ / f"{n}.png").convert("L").point(lambda v: 70 if v < 128 else 250).convert("RGB")
        im.thumbnail((cell, cell))
        x, y = (k % cols) * (cell + 24) + 12, (k // cols) * (cell + 60) + 8
        canvas.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        d.rectangle([x - 4, y - 4, x + cell + 4, y + cell + 4], outline=frame, width=4)
        d.text((x + cell // 2, y + cell + 30), lab, font=font(30, True), fill="#1F2937", anchor="mm")
    canvas.save(OUT / name)


maps([f"s00{k}" for k in range(1, 9)] + ["simple"], [f"생성 {k}" for k in range(1, 9)] + ["기본 창고"], "wh_train.png", cols=3)
maps(["fc", "s101", "s102", "s103", "s104"], ["FC (쿠팡·아마존식)", "생성 101", "생성 102", "생성 103", "생성 104"], "wh_test.png", cols=5,
     frame="#0054FF")

# ── 상상 그림: 검증 충돌 장면에서 '그대로 가면 vs 멈추면' (최종 모델) ─────────
z = np.load(V / "imag_dump.npz")
EP = "ep21150"                                                     # 왼쪽 앞에서 다가오는 물체 — 그대로 1.00, 멈춤 0.00


def big(fr, s=6, agv=True):
    im = Image.fromarray(np.kron(fr, np.ones((s, s, 1), np.uint8)))
    if agv:                                                        # 차체 직사각형 (BEV 0.15m 칸, 기준점 44행·32열)
        d = ImageDraw.Draw(im)
        c, r0 = 32 * s, 44 * s
        d.rectangle([c - 0.33 / 0.15 * s, r0 - 0.40 / 0.15 * s, c + 0.33 / 0.15 * s, r0 + 1.03 / 0.15 * s], outline="white", width=3)
    return im


big(z[f"{EP}__now"]).save(OUT / "bev_now.png")
ks = (2, 4, 6)                                                     # 0.6 · 1.0 · 1.4초 뒤
fw = 64 * 4
rows = [("실제로 일어난 일", "1.4초 뒤 충돌", "#1F2937", z[f"{EP}__real"]),
        ("상상: 그대로 가면", f"충돌 확률 {float(z[f'{EP}__p_same']):.2f}", "#DC2626", z[f"{EP}__same"]),
        ("상상: 멈추면", f"충돌 확률 {float(z[f'{EP}__p_stop']):.2f}", "#16A34A", z[f"{EP}__stop"])]
lab_w = 330
canvas = Image.new("RGB", (lab_w + len(ks) * (fw + 12), 60 + 3 * (fw + 16)), "white")
d = ImageDraw.Draw(canvas)
for c, k in enumerate(ks):
    d.text((lab_w + c * (fw + 12) + fw // 2, 30), f"{(k + 1) * 0.2:.1f}초 뒤", font=font(34, True), fill="#1F2937", anchor="mm")
for r, (t1, t2, col, frs) in enumerate(rows):
    y = 60 + r * (fw + 16)
    d.text((8, y + fw // 2 - 24), t1, font=font(32, True), fill="#1F2937", anchor="lm")
    d.text((8, y + fw // 2 + 24), t2, font=font(32, True), fill=col, anchor="lm")
    for c, k in enumerate(ks):
        canvas.paste(big(frs[min(k, len(frs) - 1)], 4), (lab_w + c * (fw + 12), y))
canvas.save(OUT / "imagine.png")

# ── 최종 시험 숫자 (1회차·2회차) · 튜닝 · 예측 정확도 → numbers.json ───────────
ok = lambda r: bool(r.get("arrived")) and not r.get("contact")    # noqa: E731


def plen(r):
    p = r.get("plan0") or []
    return sum(math.dist(a, b) for a, b in zip(p, p[1:]))


def mcnemar(rows, x, y):
    b = sum(ok(v[x]) and not ok(v[y]) for v in rows)
    c = sum(ok(v[y]) and not ok(v[x]) for v in rows)
    n = b + c
    return b, c, (min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0)


REPS = os.environ.get("REPS", "1,2").split(",")                     # 2회차가 다 끝나기 전에는 REPS=1
merged = {k: v for k, v in json.loads((V / "final/final18_merged.json").read_text()).items() if k.split("|")[1] in REPS}
full = [(k.split("|"), v) for k, v in merged.items()]
reps = sorted({k[1] for k, _ in full})
N = {"reps": len(reps), "scenes": len(full)}
for c in ("none", "rule", "AI"):
    S = [v for _, v in full if c in v]                             # 2회차 뒤쪽은 대응 없음을 뺐다
    N[c] = {"n": len(S), "arrive": sum(ok(v[c]) for v in S), "ontime": sum(ok(v[c]) and v[c]["time_s"] <= plen(v["rule"]) / 0.7 + 10 for v in S),
            "hit_obj": sum(v[c].get("hit") == "hazard" for v in S), "hit_rack": sum(v[c].get("hit") == "static" for v in S),
            "timeout": sum(not v[c].get("arrived") and not v[c].get("contact") for v in S),
            "median_s": float(np.median([v[c]["time_s"] for v in S if ok(v[c])] or [0]))}
    for typ in ("traffic", "danger", "blocked"):
        T = [v for k, v in full if k[4] == typ and c in v]
        N[c][typ] = [sum(ok(v[c]) for v in T), len(T)]
    for env in ("fc", "s101", "s102", "s103", "s104"):
        E = [v for k, v in full if k[0] == env and c in v]
        N[c][env] = [sum(ok(v[c]) for v in E), len(E)]
N["mcnemar"] = {rep: mcnemar([v for k, v in full if k[1] == rep], "rule", "AI") for rep in reps}
N["mcnemar_all"] = mcnemar([v for _, v in full], "rule", "AI")
ot = lambda v, c: ok(v[c]) and v[c]["time_s"] <= plen(v["rule"]) / 0.7 + 10   # noqa: E731
b_ = sum(ot(v, "rule") and not ot(v, "AI") for _, v in full)
c_ = sum(ot(v, "AI") and not ot(v, "rule") for _, v in full)
N["ontime_mcnemar"] = (b_, c_, min(1.0, 2 * sum(math.comb(b_ + c_, i) for i in range(min(b_, c_) + 1)) / 2 ** (b_ + c_)) if b_ + c_ else 1.0)
d_ = [v["rule"]["time_s"] - v["AI"]["time_s"] for _, v in full if ok(v["rule"]) and ok(v["AI"])]   # 둘 다 성공한 장면: 도착 시간 차
k_, m_ = sum(x > 0 for x in d_), sum(x != 0 for x in d_)
N["faster"] = {"ai_first": k_, "both": len(d_), "median_s": float(np.median(d_)) if d_ else 0.0,
               "p_sign": min(1.0, 2 * sum(math.comb(m_, i) for i in range(min(k_, m_ - k_) + 1)) / 2 ** m_) if m_ else 1.0}
same = {}
for k, v in full:
    same.setdefault((k[0],) + tuple(k[2:]), []).append(v)
for c in ("rule", "AI"):
    pairs = [x for x in same.values() if len(x) == 2]
    N[c]["repeat_same"] = [sum(ok(a[c]) == ok(b[c]) for a, b in pairs), len(pairs)]
hit_objs = {}
for _, v in full:
    if v["AI"].get("hit_obj"):
        hit_objs[v["AI"]["hit_obj"]] = hit_objs.get(v["AI"]["hit_obj"], 0) + 1
N["AI"]["hit_objs"] = hit_objs
(OUT / "numbers.json").write_text(json.dumps(N, ensure_ascii=False, indent=1))
print(json.dumps({c: {k: N[c][k] for k in ("arrive", "ontime", "hit_obj", "hit_rack", "timeout", "median_s")} for c in ("none", "rule", "AI")},
                 ensure_ascii=False), N["mcnemar"], N["mcnemar_all"], N["ontime_mcnemar"], N["faster"], N["scenes"])

# ── 비교 영상: 같은 장면 규칙(위) vs Dreamer(아래) — 원본 해상도, 한글 표시, 시간 맞춤 ───────────
SRC = V / "final/src"


def compare_video(name, rule_gif, ai_gif, rule_lab, ai_lab, t_from=12.0, t_max=36.0, poster_t=23.6):
    fr = {k: [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(SRC / g))] for k, g in (("rule", rule_gif), ("ai", ai_gif))}
    W, Hh, lab = 640, 240, 46
    out = []
    for i in range(int(t_from * 5), int(t_max * 5)):               # 기록은 5Hz
        c = Image.new("RGB", (W, 2 * (Hh + lab)), "white")
        d = ImageDraw.Draw(c)
        for r, (k, (txt, col)) in enumerate((("rule", rule_lab), ("ai", ai_lab))):
            y = r * (Hh + lab)
            f = fr[k][min(i, len(fr[k]) - 1)]
            c.paste(f, (0, y + lab))
            d.text((8, y + lab // 2), txt, font=font(26, True), fill=col, anchor="lm")
            if i >= len(fr[k]) - 1 and k == "rule":                 # 끝난 뒤(충돌)는 멈춘 화면에 표시
                d.rectangle([0, y + lab, W, y + lab + Hh], outline="#DC2626", width=6)
        d.text((W - 8, lab // 2), f"{i / 5:4.1f}초", font=font(24, True), fill="#1F2937", anchor="rm")
        out.append(c)
    to_mp4(out, name, fps=10)
    out[int((poster_t - t_from) * 5)].save(OUT / f"{name}.png")


compare_video("compare1", "s103_traffic_r0_s0_rule.gif", "s103_traffic_r0_s0_ai.gif",   # 처음 보는 창고 s103 (10-10 사용자: 더 여유 있는 예시)
              ("규칙 — 27.9초, 통로에서 나오던 사람과 충돌", "#DC2626"), ("Dreamer — 멈춰서 사람을 보낸 뒤 통과 (간격 0.38m)", "#0054FF"),
              t_from=18.0, t_max=38.0, poster_t=27.4)


# ── 발표용 짧은 영상 3개 (7분 발표 — 합쳐 30초 안쪽) ─────────────────────────
def gif_frames(path):
    return [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(path))]


def mp4_frames(path):
    """mp4 → PIL 프레임 (ffmpeg로 풀기)."""
    w, h = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                    "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    return [Image.frombytes("RGB", (w, h), raw[i * w * h * 3:(i + 1) * w * h * 3]) for i in range(n)]


# ① 문제 제기: 규칙 AGV가 랙 사이에서 나온 사람과 부딪히는 5초 (뒤따르는 카메라, 2배 확대)
rule = gif_frames(SRC / "s103_traffic_r0_s0_rule.gif")
clip = []
for i in range(105, len(rule)):
    f = rule[i].crop((320, 0, 640, 240)).resize((640, 480), Image.BICUBIC)
    d = ImageDraw.Draw(f)
    d.rectangle([0, 0, 640, 44], fill="white")
    d.text((10, 22), "규칙 AGV — 가까이 보이면 멈춘다", font=font(26, True), fill="#1F2937", anchor="lm")
    if i == len(rule) - 1:
        d.rectangle([0, 44, 639, 479], outline="#DC2626", width=8)
        d.text((620, 22), "충돌", font=font(28, True), fill="#DC2626", anchor="rm")
    clip.append(f)
to_mp4(clip + [clip[-1]] * 6, "v1_rule_hit", fps=6)
clip[-1].save(OUT / "v1_rule_hit.png")

# ② Cosmos: 실제(부딪힘) | Cosmos 상상(통이 물러남) 나란히, 두 번 반복
real, imag = mp4_frames(OLD / "s6_real_straight.mp4"), mp4_frames(OLD / "s6_cosmos_straight.mp4")
n = max(len(real), 2 * len(imag))
pair = []
for i in range(n):
    a = real[min(i * len(real) // n, len(real) - 1)].resize((480, 360))
    b = imag[min(i * len(imag) // n, len(imag) - 1)].resize((480, 360))
    c = Image.new("RGB", (980, 410), "white")
    c.paste(a, (0, 50))
    c.paste(b, (500, 50))
    d = ImageDraw.Draw(c)
    d.text((240, 25), "실제 — 그대로 가면 부딪힘", font=font(26, True), fill="#DC2626", anchor="mm")
    d.text((740, 25), "Cosmos의 상상 — 통이 같이 물러남", font=font(26, True), fill="#0054FF", anchor="mm")
    pair.append(c)
to_mp4(pair * 2, "v2_cosmos", fps=15)
pair[int(len(pair) * 0.6)].save(OUT / "v2_cosmos.png")
print("그림·영상 →", OUT)


# ── 길찾기 그림: 시험 창고 s101 — 처음 지도(랙·벽만) 위 A* 경로, 통로가 막히면 다시 찾은 경로 ─────────
import sys                                                         # noqa: E402
sys.path.insert(0, "/data/ch/agv-wm/isaac")
import nav                                                         # noqa: E402

D101 = R / "wh/s101"
meta = json.loads((D101 / "inspect.json").read_text())
E101 = json.loads((D101 / "env.json").read_text())
occ = np.load(D101 / "grid_racks.npy")
gm = nav.GridMap(occ, meta["cell"], meta["x0"], meta["y0"])
start = tuple(E101["entrance"])


def inside(al, q, m=0.0):
    r_ = al["rect"]
    return r_[0] - m <= q[0] <= r_[2] + m and r_[1] - m <= q[1] <= r_[3] + m


best = None
for gname, goal in sorted(E101["slots"].items()):
    p0 = nav.plan(gm.coarse(0.1), start, tuple(goal), robot_r=0.7, soft_r=1.5)
    if p0 is None:
        continue
    al = next(a for a in E101["aisles"] if inside(a, goal))
    k_in = next(i for i, q in enumerate(p0) if inside(al, q))     # 경로가 그 통로에 처음 들어가는 점
    mouth = min(al["mouths"], key=lambda m_: math.dist(m_, p0[k_in]))
    ax = al["axis"] == "y"
    r_ = al["rect"]
    inward = (0.0, math.copysign(1, (r_[1] + r_[3]) / 2 - mouth[1])) if ax else (math.copysign(1, (r_[0] + r_[2]) / 2 - mouth[0]), 0.0)
    c_ = (mouth[0] + inward[0] * 1.6, mouth[1] + inward[1] * 1.6)
    blk = occ.copy()                                               # 통로 폭 전체를 팔레트(1.2m 두께)로 막음
    x0_, x1_ = (r_[0], r_[2]) if ax else (c_[0] - 0.6, c_[0] + 0.6)
    y0_, y1_ = (c_[1] - 0.6, c_[1] + 0.6) if ax else (r_[1], r_[3])
    i0, i1 = int((y0_ - gm.y0) / gm.cell), int((y1_ - gm.y0) / gm.cell) + 1
    j0, j1 = int((x0_ - gm.x0) / gm.cell), int((x1_ - gm.x0) / gm.cell) + 1
    blk[max(i0, 0):i1, max(j0, 0):j1] = True
    dense = nav.densify(p0, 0.1)                                   # 처음 길을 따라가다 막힘이 3m 앞에 보이는 자리
    near = lambda q: math.hypot(max(x0_ - q[0], 0, q[0] - x1_), max(y0_ - q[1], 0, q[1] - y1_))   # noqa: E731
    k_see = next((i for i, q in enumerate(dense) if near(q) <= 3.0), None)
    if k_see is None or k_see < 10:
        continue
    seen = tuple(dense[k_see])
    p1 = nav.plan(nav.GridMap(blk, gm.cell, gm.x0, gm.y0).coarse(0.1), seen, tuple(goal), robot_r=0.7, soft_r=1.5)
    if p1 is None:
        continue
    L1 = sum(math.dist(a, b) for a, b in zip(p1, p1[1:]))
    if best is None or L1 > best[0]:
        best = (L1, gname, goal, p0, p1, (x0_, y0_, x1_, y1_), seen, dense[:k_see + 1])
_, gname, goal, p0, p1, box_, seen, walked = best
z_ = 4                                                             # 지도 칸 → 그림 픽셀
img = Image.fromarray(np.where(occ[::-1], 60, 247).astype(np.uint8)).convert("RGB")
img = img.resize((img.width * z_ // 2, img.height * z_ // 2), Image.NEAREST)
sc = z_ / 2 / gm.cell
H_ = img.height
P = lambda q: ((q[0] - gm.x0) * sc, H_ - (q[1] - gm.y0) * sc)    # noqa: E731
d = ImageDraw.Draw(img)
rest = nav.densify(p0, 0.1)[len(walked) - 1:]                      # 버린 나머지 길 — 점선
for i in range(0, len(rest) - 4, 8):
    d.line([P(rest[i]), P(rest[i + 4])], fill="#93C5FD", width=6)
d.line([P(q) for q in walked], fill="#0054FF", width=8)          # 실제로 간 길
d.line([P(q) for q in p1], fill="#16A34A", width=8)              # 다시 찾은 길
d.rectangle([P((box_[0], box_[3])), P((box_[2], box_[1]))], fill="#F59E0B")
x, y = P(seen)
d.ellipse([x - 16, y - 16, x + 16, y + 16], fill="white", outline="#DC2626", width=6)
d.text((x + 24, y - 24), "막힘 발견 → 여기서 다시 찾기", font=font(34, True), fill="#DC2626", anchor="lb")
for q, col, lab, anc, off in ((start, "#1F2937", "입구", "mt", (0, 22)), (tuple(goal), "#DC2626", f"목적지 {gname}", "mb", (0, -20))):
    x, y = P(q)
    d.ellipse([x - 14, y - 14, x + 14, y + 14], fill=col)
    d.text((x + off[0], y + off[1]), lab, font=font(36, True), fill=col, anchor=anc)
pts = [P(q) for q in list(p0) + list(p1)] + [P((box_[0], box_[1])), P((box_[2], box_[3]))]
xs, ys = [q[0] for q in pts], [q[1] for q in pts]
m_ = 7.0 * sc                                                      # 둘레 7m
img.crop((max(0, min(xs) - m_), max(0, min(ys) - m_), min(img.width, max(xs) + 1.5 * sc), min(img.height, max(ys) + m_))).save(OUT / "nav_s101.png")
print("길찾기 그림:", gname, "처음 길", round(sum(math.dist(a, b) for a, b in zip(p0, p0[1:])), 1), "m · 발견 지점에서 새 길",
      round(best[0], 1), "m")
for src, dst in ((OLD8 / "s10_purepursuit.png", "purepursuit.png"), (OLD8 / "s5_agv.png", "agv.png")):
    shutil.copy(src, OUT / dst)


# ── 학습 데이터: 3인칭 · 깊이 카메라 원래 화면 · 위에서 본 2D 지도 (FC, 사람 가로지름, 0.2초 한 칸) ─────────────
import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
from matplotlib import cm                                          # noqa: E402
zv = np.load(V / "demo/fc_crossing_vis.npz")
K_ = 140
Image.fromarray(zv["view"][K_][:, 320:]).resize((640, 480), Image.BICUBIC).save(OUT / "data_view.png")
dep = np.nan_to_num(zv["depth"][K_].astype(np.float32), nan=10.0, posinf=10.0)
col = (cm.get_cmap("turbo")(1 - np.clip(dep / 10.0, 0, 1))[..., :3] * 255).astype(np.uint8)   # 가까울수록 빨강
Image.fromarray(col).resize((640, 480), Image.BILINEAR).save(OUT / "data_depth.png")
b = zv["bev"][K_].copy()
b[..., 0] = 0                                                      # 모델 입력과 같게 (지도 채널 끔)
big(b, 7).save(OUT / "data_bev.png")
print("학습 데이터 칸", K_, "행동(v, w)", zv["act"][K_].round(2).tolist(), "간격", round(float(zv["gap"][K_]), 2))
zb = np.load(V / "demo/bev_variety.npz")
names, bevs = list(zb["names"]), zb["bevs"]
cell = 64 * 4
canvas = Image.new("RGB", (len(names) * (cell + 16), cell + 56), "white")
d = ImageDraw.Draw(canvas)
for i, (nm, bv) in enumerate(zip(names, bevs)):
    canvas.paste(big(bv, 4), (i * (cell + 16), 0))
    d.text((i * (cell + 16) + cell // 2, cell + 28), str(nm), font=font(26, True), fill="#1F2937", anchor="mm")
canvas.save(OUT / "bev_variety.png")

# ── 공식 이미지 (NVIDIA Cosmos · Isaac Sim, DreamerV3 논문 그림, idealworks iw.hub) ───────────────
WEB = OUT / "web"
Image.open(WEB / "cosmos_header.png").convert("RGB").save(OUT / "cosmos_logo.png")
Image.open(WEB / "cosmos_diagram.png").convert("RGB").save(OUT / "cosmos_diagram.png")
Image.open(WEB / "isaacsim_hero.png").convert("RGB").save(OUT / "isaacsim_hero.png")
dm = Image.open(WEB / "dreamer_method.png").convert("RGB")
dm.crop((0, 0, int(dm.width * 0.53), dm.height)).save(OUT / "dreamer_wm.png")  # (a) World Model Learning 부분만
tg = Image.open(WEB / "dreamer_tasks.gif")
tg.seek(0)
tg.convert("RGB").save(OUT / "dreamer_tasks.png")
at = Image.open(WEB / "agv_top.png").convert("RGBA")
bg = Image.new("RGBA", at.size, (255, 255, 255, 255))
Image.alpha_composite(bg, at).convert("RGB").save(OUT / "agv_real.png")
print("공식 이미지 손질 끝")


# ── 위에서 내려다본 비교 영상 (규칙 왼쪽 · Dreamer 오른쪽) — 10-10 사용자: 충돌·회피가 분명하게 구분되는 영상 ─────────
def compare_top(name, rule_gif, ai_gif, rule_lab, ai_lab, t_from, t_to, poster_t, fps=8, keep=0.6, center=(0.5, 0.5)):
    fr = {k: [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(SRC / g))] for k, g in (("rule", rule_gif), ("ai", ai_gif))}
    S_, lab, gap_ = 560, 54, 16
    out = []
    for i in range(int(t_from * 5), int(t_to * 5)):
        c = Image.new("RGB", (2 * S_ + gap_, S_ + lab), "white")
        d = ImageDraw.Draw(c)
        for col_, (k, (txt, colr)) in enumerate((("rule", rule_lab), ("ai", ai_lab))):
            x0 = col_ * (S_ + gap_)
            f = fr[k][min(i, len(fr[k]) - 1)]
            hw = f.width * keep / 2                                    # 보고 싶은 곳(center, 비율)만 크게 (keep 비율 창)
            cx = min(max(center[0] * f.width, hw), f.width - hw)
            cy = min(max(center[1] * f.height, hw), f.height - hw)
            f = f.crop((int(cx - hw), int(cy - hw), int(cx + hw), int(cy + hw))).resize((S_, S_), Image.BICUBIC)
            c.paste(f, (x0, lab))
            d.text((x0 + 8, lab // 2), txt, font=font(25, True), fill=colr, anchor="lm")
            if k == "rule" and i >= len(fr[k]) - 1:                    # 충돌로 끝난 뒤: 멈춘 화면 + 빨간 테두리
                d.rectangle([x0, lab, x0 + S_ - 1, lab + S_ - 1], outline="#DC2626", width=10)
                d.text((x0 + S_ // 2, lab + 40), "충돌!", font=font(46, True), fill="#DC2626", anchor="mm")
        d.text((2 * S_ + gap_ - 8, lab // 2), f"{i / 5:4.1f}초", font=font(24, True), fill="#1F2937", anchor="rm")
        out.append(c)
    to_mp4(out, name, fps=fps)
    out[min(len(out) - 1, int((poster_t - t_from) * 5))].save(OUT / f"{name}.png")
    return out


compare_top("compare_top", "top_fc_01_rule.gif", "top_fc_01_ai.gif",
            ("규칙 — 23.0초, 사람과 충돌", "#DC2626"), ("Dreamer — 사람을 피해 지나감 → 도착", "#0054FF"),
            t_from=15.0, t_to=30.0, poster_t=22.8)
print("위에서 본 비교 영상 → compare_top.mp4")

# ── 6장용: 같은 장면 규칙만 (위에서 본 화면) — 충돌까지 ────────────────────────────
rule_top = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(SRC / "top_fc_01_rule.gif"))]
clip = []
for i in range(75, len(rule_top)):
    f = rule_top[i]
    m_ = int(f.width * 0.2)
    f = f.crop((m_, m_, f.width - m_, f.height - m_)).resize((600, 600), Image.BICUBIC)
    c = Image.new("RGB", (600, 650), "white")
    c.paste(f, (0, 50))
    d = ImageDraw.Draw(c)
    d.text((10, 25), "규칙 AGV — 가까이 보이면 멈춘다", font=font(26, True), fill="#1F2937", anchor="lm")
    d.text((590, 25), f"{i / 5:4.1f}초", font=font(24, True), fill="#1F2937", anchor="rm")
    if i == len(rule_top) - 1:
        d.rectangle([0, 50, 599, 649], outline="#DC2626", width=10)
        d.text((300, 92), "충돌!", font=font(46, True), fill="#DC2626", anchor="mm")
    clip.append(c)
to_mp4(clip + [clip[-1]] * 10, "rule_top", fps=6)
clip[-1].save(OUT / "rule_top.png")
print("6장용 규칙 영상 → rule_top.mp4")


# ── 9장 최종: 바닥 고정 카메라(두 주행이 같은 화면) — s102 '건너다 멈칫하는 사람', 같은 주행에서 규칙 충돌 · Dreamer 통과 ─────
compare_top("compare_fixed", "fx102_h6_rule.gif", "fx102_h6_ai.gif",
            ("규칙 — 감속하다 14.9초에 사람과 충돌", "#DC2626"), ("Dreamer — 사람보다 먼저 안전하게 통과", "#0054FF"),
            t_from=9.0, t_to=19.0, poster_t=14.6, fps=5, keep=0.82, center=(0.50, 0.58))
print("고정 카메라 비교 영상 → compare_fixed.mp4")

# ── 6장용 (9장과 같은 주행·같은 고정 카메라): 규칙만 — 충돌까지 ──────────────────────
rf = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(SRC / "fx102_h6_rule.gif"))]
clip = []
for i in range(45, len(rf)):
    f = rf[i]
    hw = f.width * 0.82 / 2
    cx, cy = min(max(0.50 * f.width, hw), f.width - hw), min(max(0.58 * f.height, hw), f.height - hw)
    f = f.crop((int(cx - hw), int(cy - hw), int(cx + hw), int(cy + hw))).resize((600, 600), Image.BICUBIC)
    c = Image.new("RGB", (600, 650), "white")
    c.paste(f, (0, 50))
    d = ImageDraw.Draw(c)
    d.text((10, 25), "규칙 AGV — 가까이 보이면 감속·정지", font=font(26, True), fill="#1F2937", anchor="lm")
    d.text((590, 25), f"{i / 5:4.1f}초", font=font(24, True), fill="#1F2937", anchor="rm")
    if i == len(rf) - 1:
        d.rectangle([0, 50, 599, 649], outline="#DC2626", width=10)
        d.text((300, 92), "충돌!", font=font(46, True), fill="#DC2626", anchor="mm")
    clip.append(c)
to_mp4(clip + [clip[-1]] * 8, "rule_fixed", fps=5)
clip[-1].save(OUT / "rule_fixed.png")
print("6장용 규칙 영상(고정 카메라) → rule_fixed.mp4")
