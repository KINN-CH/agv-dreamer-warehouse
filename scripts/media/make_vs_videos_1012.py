#!/usr/bin/env python3
"""10/12 발표용 '규칙은 부딪히고 Dreamer는 피하는' 비교 영상 (10-11 사용자 요청: 청중이 신기하다고 느낄 영상 3개).
같은 장면·같은 바닥 고정 카메라에서 규칙(왼쪽)과 Dreamer(오른쪽) — VM ~/agv/fix2.sh 결과(GIF·results.json)를 받아 합친다.
make_media_1012.compare_top을 옮겨 왔다 (그 파일은 실행하면 모든 그림을 다시 만들어서 import하지 않는다).

    python3 docs/make_vs_videos_1012.py <이름> <규칙.gif> <Dreamer.gif> <results.json> "<규칙 설명>" "<Dreamer 설명>" [keep] [cx,cy]
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

OUT = Path("/data/ch/agv-wm/results/v5c/videos/vs1012")
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size):
    return ImageFont.truetype(FONT_B, size, index=1)


def to_mp4(frames, path, fps):
    w, h = frames[0].size
    w2, h2 = w // 2 * 2, h // 2 * 2
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w2}x{h2}", "-r", str(fps),
                          "-i", "-", "-pix_fmt", "yuv420p", "-crf", "18", str(path)], stdin=subprocess.PIPE)
    for f in frames + [frames[-1]] * fps * 2:                       # 마지막 장면 2초 멈춤
        p.stdin.write(np.asarray(f.convert("RGB").crop((0, 0, w2, h2))).tobytes())
    p.stdin.close()
    p.wait()


def main(name, rule_gif, ai_gif, res_json, rule_lab, ai_lab, keep=0.95, center=(0.5, 0.5), fps=5):
    fr = {k: [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(g))] for k, g in (("rule", rule_gif), ("ai", ai_gif))}
    s = json.load(open(res_json))[0]
    rule, ai = s["respond"], s["wm"]
    t_hit = rule["time_s"]
    hx, hy = rule["traj"][-1]                                        # 규칙이 부딪힌 자리
    d = [math.hypot(x - hx, y - hy) for x, y in ai["traj"]]
    t_pass = ai["time_s"] * int(np.argmin(d)) / (len(d) - 1)         # Dreamer가 그 자리를 지나간 때 (traj는 0.33초 간격)
    t_from, t_to = max(0.0, t_hit - 8.0), max(t_hit, t_pass) + 3.0
    S_, lab, gap_ = 600, 58, 16
    out = []
    for i in range(int(t_from * 5), int(t_to * 5)):                  # GIF = 0.2초마다 한 장
        c = Image.new("RGB", (2 * S_ + gap_, S_ + lab), "white")
        dr = ImageDraw.Draw(c)
        for col, (k, (txt, colr)) in enumerate((("rule", (rule_lab, "#DC2626")), ("ai", (ai_lab, "#0054FF")))):
            x0 = col * (S_ + gap_)
            f = fr[k][min(i, len(fr[k]) - 1)]
            hw = f.width * keep / 2                                  # 보고 싶은 곳(center, 비율)만 크게
            cx = min(max(center[0] * f.width, hw), f.width - hw)
            cy = min(max(center[1] * f.height, hw), f.height - hw)
            c.paste(f.crop((int(cx - hw), int(cy - hw), int(cx + hw), int(cy + hw))).resize((S_, S_), Image.BICUBIC), (x0, lab))
            dr.text((x0 + 8, lab // 2), txt, font=font(25), fill=colr, anchor="lm")
            if k == "rule" and i >= len(fr[k]) - 1:                  # 충돌로 끝난 뒤: 멈춘 화면 + 빨간 테두리
                dr.rectangle([x0, lab, x0 + S_ - 1, lab + S_ - 1], outline="#DC2626", width=10)
                dr.text((x0 + S_ // 2, lab + 44), "충돌!", font=font(48), fill="#DC2626", anchor="mm")
        dr.text((2 * S_ + gap_ - 8, lab // 2), f"{i / 5:4.1f}초", font=font(24), fill="#1F2937", anchor="rm")
        out.append(c)
    OUT.mkdir(parents=True, exist_ok=True)
    to_mp4(out, OUT / f"{name}.mp4", fps)
    out[min(len(out) - 1, int((t_hit - t_from) * 5) - 1)].save(OUT / f"{name}.png")   # 충돌 직전 장면
    print(f"{name}: {t_from:.1f}~{t_to:.1f}초 · 규칙 충돌 {t_hit}초 · Dreamer 그 자리 통과 {t_pass:.1f}초 → {OUT / name}.mp4")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*a[:6], keep=float(a[6]) if len(a) > 6 else 0.95,
         center=tuple(map(float, a[7].split(","))) if len(a) > 7 else (0.5, 0.5))
