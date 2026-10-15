# Isaac Sim 실행 환경 (RTX 5090 VM)

`gen_episodes.py`가 AGV 접근 에피소드를 만들어 `docs/02-데이터규격.md` 형식으로 내놓는다.

## 환경 (2026-09-23 구성 완료)

| 항목 | 값 |
|---|---|
| 접속 | `ssh <user>@<VM_IP>` · Tailscale `<TAILSCALE_IP>` (호스트명 `<host>`) |
| GPU | RTX 5090 32GB, PCIe 패스스루 (RT Core 정상) |
| Isaac Sim | **6.1** pip 설치, venv `~/isaacsim-venv` |
| 작업 폴더 | `~/agv/isaac/` (이 폴더의 사본) |

## 에피소드 생성

```bash
ssh <user>@<VM_IP>
cd ~/agv/isaac && source ~/isaacsim-venv/bin/activate
python gen_episodes.py --scenes 3 --out ./out      # 예행연습
python gen_episodes.py --scenes 50 --out ./out     # 본실험 (장면당 직진·정지 × 장애물·빈 장면)
python gen_episodes.py --calib --out ./calib       # 전진 보정용 빈 장면 직진 3초 (calibrate.py)
```

AGX로 옮기기 (AGX에서 실행):

```bash
rm -rf /data/ch/agv-wm/data/incoming/*
scp -r <user>@<VM_IP>:~/agv/isaac/out/ep_* /data/ch/agv-wm/data/incoming/
```

## 장면 구성

- **환경**: Simple_Warehouse
- **AGV**: idealworks iw.hub — 실제 창고용 AGV. 치수(바퀴 반지름 0.081, 바퀴 간격 0.579, 차체 앞 0.398·뒤 −1.033·반폭 0.33)는 에셋에서 실측
- **장애물**: 창고 소품 5종을 장면마다 무작위 — 종이박스 A·B, 드럼통 A·B·C. **카메라(0.25m)보다 낮은 소품은 뺐다** (화면에서 바닥에 묻힌다)
- **카메라**: 차체 앞면보다 3.8cm 안쪽, 높이 0.25m, 화각 60°
- **명령**: 조건 프레임 전 0.5초는 직진, 이후 2초 동안 `straight`(1m/s 직진) 또는 `brake`(1m/s² 감속, 1초에 0.5m 가서 정지)
- **빈 장면 판**: 에피소드마다 같은 자리에서 장애물만 치우고 한 번 더 달린다 → `cond_empty.png`, `future_empty.mp4`
- **충돌 판정**: 차체 직사각형과 장애물 직사각형 사이 거리가 0. 닿으면 비상 정지

## 초기 실행 시 문제 및 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| Jetbot을 못 찾음 | 6.1에서 에셋 경로 변경 | `Robots/NVIDIA/Jetbot`, `Robots/Idealworks/iwhub`로 교체 |
| core/sensors 임포트 실패 | 6.1에서 deprecated 폴더로 이동 | `enable_extension`으로 명시적으로 켬 |
| 직진해도 충돌 안 함 | 장애물이 2초 거리보다 멀었음 | 앞면까지 1.0~1.6m |
| **충돌 순간 화면이 검다** | ① 카메라가 차체 앞에 있어 장애물 안으로 들어감 ② 계속 밀어서 파고듦 ③ 근접 클리핑 | ① 카메라를 차체 안쪽으로 ② 닿으면 정지 ③ 클리핑 5mm |
| 단색 상자라 채점기가 흐름을 못 잡음 | 광학흐름은 무늬를 따라간다 | 실제 창고 소품으로 교체 |

## 원격 화면 스트리밍

```bash
~/agv/stream.sh     # VM에서. "app ready"가 뜨면 접속
```

맥북 **Isaac Sim WebRTC Streaming Client**(aarch64) → `<TAILSCALE_IP>`, signal 49100, stream 47998.

**반드시 Tailscale 주소로 접속한다.** 연구실 망에서 AGX·맥북 쪽 대역과 VM 대역 사이 장비가
SSH만 통과시키고 스트리밍 포트를 막아서, `<VM_IP>`로 붙으면 검은 화면만 뜬다.
Isaac Sim은 한 번에 하나만 띄운다 — 두 개면 스트리밍 포트가 겹쳐 역시 검은 화면이 된다.
