#!/bin/bash
# 직접 해 보기 — 창고·출발·목적지·상황을 골라 최종 Dreamer(v5c, 설정 C)로 한 번 주행.
# 맥북: ssh <user>@<TAILSCALE_IP> → ~/agv/demo.sh → (실시간 보기면) Isaac Sim WebRTC Streaming Client로 <TAILSCALE_IP> 접속
cd ~/agv/isaac && source ~/isaacsim-venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export AGV_STREAM_IP=<TAILSCALE_IP>   # 맥북 실시간 보기 (Tailscale)
if ps -eo cmd | grep -v grep | grep -q "logistics.py.*--stream\|isaacsim.exp.full.streaming"; then
    echo "이미 실시간 보기 Isaac이 돌고 있습니다 (스트림은 하나만 가능). 끝난 뒤 다시 실행하세요."; exit 1
fi
echo "창고 — 학습한 창고: s001 s002 s003 s004 s005 s006 s007 s008 simple"
echo "       처음 보는 창고: fc(쿠팡·아마존식) s101 s102 s103 s104"
read -p "창고 [s101]: " W; W=${W:-s101}
case $W in
  fc) D=~/agv/fc; E=$D/fc_env.json ;;
  simple) D=~/agv/logistics; E=$D/simple_env.json ;;
  *) D=~/agv/wh/$W; E=$D/env.json ;;
esac
[ -f "$E" ] || { echo "창고 '$W'가 없습니다"; exit 1; }
python3 - "$E" <<'EOF'
import json, sys, collections
E = json.load(open(sys.argv[1]))
by = collections.defaultdict(list)
for k in sorted(E["slots"]):
    by[k.rsplit("-", 1)[0]].append(k.rsplit("-", 1)[1])
print("칸 (통로: 번호):")
for a, ns in sorted(by.items()):
    print(f"  {a}: {' '.join(ns)}")
EOF
read -p "출발 ('입구' 또는 칸 이름, 예 A1-01) [입구]: " S; S=${S:-입구}
read -p "목적지 칸 (예 A7-02): " G
[ -n "$G" ] || { echo "목적지를 입력하세요"; exit 1; }
read -p "상황 — traffic(평소 통행) / danger(갑자기 덮침) / blocked(통로 막힘) [danger]: " T; T=${T:-danger}
read -p "무작위 배치 번호 (같은 번호 = 같은 장면) [1]: " N; N=${N:-1}
read -p "주행 — wm(Dreamer만) / both(규칙 → Dreamer 차례로) [wm]: " C; C=${C:-wm}
[ "$C" = both ] && C=respond,wm
read -p "실시간 보기 (맥북 Streaming Client) y/n [y]: " L; L=${L:-y}
X=""; [ "$L" = y ] && X="--stream --wait 40" && echo ">> 'N초 뒤 출발'이 뜨면 맥북 Streaming Client로 <TAILSCALE_IP> 접속하세요"
echo ">> Isaac Sim 시작 중 (1~2분)..."
python logistics.py --v5 --demo --env $E --out $D --map racks --start "$S" --goal "$G" --types $T --demo-seed $N \
    --backout --swept --smooth --real-motion --conds $C --wm-ckpt $(cat ~/agv/wm_bev_v5c.best) \
    --hybrid --pathcand --escape --hyb-slow 2.5:0.6 --tau 0.05 $X 2>&1 | grep --line-buffered "장면:\|초 뒤 출발\|결과\|영상:\|칸 .\|길이 없\|Traceback"
echo "영상 폴더: $D/demo/  (맥북으로 받기: scp '<user>@<TAILSCALE_IP>:$D/demo/*.mp4' .)"
