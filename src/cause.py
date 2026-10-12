#!/usr/bin/env python3
"""원인 확인 — 장애물 앞 직진에서 충돌이 안 나오는 건 설정 탓인가, 모델 탓인가.

본실험(2×2)에서 명령대로 안 된 칸은 '장애물 + 직진' 하나다(0/50). 그 칸만 떼어 설정을 하나씩 바꿔 다시 상상시킨다.
빈 장면에도 같은 설정으로 직진을 시켜, 바꾼 설정에서도 Cosmos가 명령대로 달리는지 같이 본다(대조).

    기준  본실험 그대로 — 전진 액션 0.83(1m/s), seed 0, 장면 묘사 한 문장. 이미 돌린 영상을 쓴다
    ①    전진 액션 4 — 보정표에서 상상 속 전진이 가장 컸던 값 (0.093m/스텝 ≈ 1.4m/s)
    ②    seed 1, 2
    ③    장면 묘사에서 'empty' 한 단어를 뺀다
    ④    ③ 문장 끝에 "앞의 물체에 부딪힌다"를 쓴다 — 모델이 충돌을 그릴 수는 있는지 본다. 빈 장면 대조는 없다
    ⑤    문장에 "AGV 시점이고, 보이는 것은 실제 환경이라 모든 물체가 멈춰 있다"를 쓴다 (9/28 추가)

기준은 본실험과 같다 — 상상이 같은 장면의 실제 결말 두 개(직진 = 충돌 / 정지) 중 어느 쪽과 더 닮았나.
장애물 칸은 배경만 비교(background.py)도 같이 본다 — 충돌은 없어도 카메라는 직진만큼 갔나.
장면은 장애물 5종 × 2개 (번호 순으로 앞에서 둘).

    agvrun src/cause.py
결과: results/cause/summary.txt, result.csv, 장면별 비교 그림 scene_*.png
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import actions as A          # noqa: E402
import background as B       # noqa: E402
import experiment as E       # noqa: E402
import rollout as R          # noqa: E402
import scoring as S          # noqa: E402

OUT, ROLL = E.OUT / "cause", E.ROLL / "cause"
PER_KIND = 2
BASE = "기준 (본실험)"
NO_EMPTY = R.PROMPT.replace("large, empty indoor", "large indoor")
CRASH = NO_EMPTY + " The robot drives straight ahead and crashes into the object in front of it."
STATIC = ("Front camera view of an AGV (automated guided vehicle) in a large indoor warehouse with white brick walls, "
          "yellow upper walls and a blue-gray concrete floor. This is a real, static environment: "
          "every object on the floor stays still and never moves.")
assert NO_EMPTY != R.PROMPT
# 이름: (표에 쓸 설명, 전진 액션 값(None = 기준 1m/s), seed, 문장, 빈 장면 대조) — 이 순서로 돌린다
VARIANTS = {
    "static":  ("⑤ 문장: AGV 시점 + 물체는 멈춰 있음", None, 0, STATIC, True),
    "crash":   ("④ 문장으로 충돌을 말함", None, 0, CRASH, False),
    "action4": ("① 전진 액션 0.83→4", 4.0, 0, R.PROMPT, True),
    "noempty": ("③ 문장에서 'empty' 뺌", None, 0, NO_EMPTY, True),
    "seed1":   ("② seed 1", None, 1, R.PROMPT, True),
    "seed2":   ("② seed 2", None, 2, R.PROMPT, True),
}


def pick_scenes() -> list[tuple[str, dict]]:
    """장애물 종류마다 번호 순으로 앞에서 PER_KIND개."""
    kinds: dict[str, list] = {}
    for sid, eps in sorted(E.scene_pairs().items()):
        kind = json.loads((eps["straight"] / "meta.json").read_text())["obstacle"]
        if len(kinds.setdefault(kind, [])) < PER_KIND:
            kinds[kind].append((sid, eps))
    return sorted(x for v in kinds.values() for x in v)


def summarize(rows: list[dict], n: int) -> str:
    L = [f"원인 확인 — 장면 {n}개. 칸마다 장면 수", "",
         "설정 | 장애물: 충돌을 그림 | 장애물: 카메라는 직진만큼 감(배경만 비교) | 빈 장면: 직진 따름(대조)"]
    for lab in [BASE] + sorted(v[0] for v in VARIANTS.values()):
        pick = lambda s, k: [r[k] for r in rows if r["설정"] == lab and r["장면"] == s]
        ob, cam, em = pick("장애물", "직진 결말을 닮음"), pick("장애물", "카메라는 직진만큼 감"), pick("빈 장면", "직진 결말을 닮음")
        L.append(f"{lab} | {sum(ob)}/{len(ob)} | {sum(cam)}/{len(cam)} | " + (f"{sum(em)}/{len(em)}" if em else "—"))
    L += ["", "충돌 칸이 올라가면 그 설정이 원인이다. 카메라 칸은 장애물 자리를 가리고 배경만 빈 장면의 실제 직진·정지와",
          "비교한다(background.py와 같은 방법). 빈 장면 칸은 바꾼 설정에서도 Cosmos가 직진 명령을 따르는지 본다.",
          "주의 ① — 카메라가 실제 1m/s보다 멀리 가서 자(실제 직진·정지 영상)의 범위 밖이다. 카메라 칸을 쓰지 않는다(그림으로 본다).",
          "주의 ② seed 2 — 빈 장면에서도 직진을 따르지 않았다(대조 실패). 이 줄은 원인 판단에 쓰지 않는다."]
    return "\n".join(L)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ROLL.mkdir(parents=True, exist_ok=True)
    scenes = pick_scenes()
    print(f"장면 {len(scenes)}개: {', '.join(s for s, _ in scenes)}", flush=True)
    real = {(sid, sk): {c: E.truth(eps[c], cond, future) for c in E.COMMANDS}
            for sid, eps in scenes for sk, (_, cond, future, _) in E.SCENES.items()}
    keep = {sid: B.keep_mask(eps["straight"]) for sid, eps in scenes}
    rows, imags, pipe = [], {}, None

    def score(sid: str, sk: str, key: str, label: str, vid: Path) -> None:
        imag = S.read_frames(vid)
        d = {c: S.distance(imag, real[(sid, sk)][c]) for c in E.COMMANDS}
        cam = ""
        if sk == "obstacle":                       # 장애물을 가리고 배경만 빈 장면의 실제 결말과 비교
            imags[(sid, key)] = imag
            bg = {c: S.distance(imag, real[(sid, "empty")][c], keep[sid]) for c in E.COMMANDS}
            cam = bg["straight"] < bg["brake"]
        rows.append({"scene": sid, "설정": label, "장면": E.SCENES[sk][0], "직진 결말을 닮음": d["straight"] < d["brake"],
                     "직진 정답과 차이": round(d["straight"], 3), "정지 정답과 차이": round(d["brake"], 3),
                     "카메라는 직진만큼 감": cam})
        print(f"  {label:18s} 장면 {sid} {E.SCENES[sk][0]:4s} → "
              f"{'직진' if d['straight'] < d['brake'] else '정지'} 결말과 닮음 "
              f"(차이: 직진 {d['straight']:.3f} · 정지 {d['brake']:.3f})"
              + (f" · 배경: 카메라 {'직진만큼 감' if cam else '덜 감'}" if sk == "obstacle" else ""), flush=True)

    for sid, eps in scenes:
        for sk, (_, _, _, suffix) in E.SCENES.items():
            score(sid, sk, "base", BASE, E.ROLL / f"{eps['straight'].name}{suffix}.mp4")
    for key, (label, tz, seed, prompt, control) in VARIANTS.items():
        for sid, eps in scenes:
            ep = eps["straight"]
            meta = json.loads((ep / "meta.json").read_text())
            action = A.forward([tz] * E.CHUNK) if tz else E.command_action(meta)
            for sk, (_, cond, _, suffix) in E.SCENES.items():
                if sk == "empty" and not control:
                    continue
                vid = ROLL / f"{ep.name}{suffix}_{key}.mp4"
                if not vid.exists():
                    pipe = pipe or R.build_pipeline()
                    fr, _, _ = R.rollout(pipe, R.load_frame(ep / cond), action, seed=seed, prompt=prompt)
                    R.save_video(fr, vid, fps=round(meta["fps"] / E.STRIDE))
                score(sid, sk, key, label, vid)

    for sid, _ in scenes:
        r = real[(sid, "obstacle")]
        E.draw_rows([("real straight (crash)", r["straight"]), ("real brake", r["brake"]),
                     ("Cosmos: baseline", imags[(sid, "base")])]
                    + [(f"Cosmos: {k}", imags[(sid, k)]) for k in VARIANTS], OUT / f"scene_{sid}.png")
    with open(OUT / "result.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, len(scenes))
    (OUT / "summary.txt").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
