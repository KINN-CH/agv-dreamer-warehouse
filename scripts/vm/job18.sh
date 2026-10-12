#!/bin/bash
# 평가 한 작업 (v5c 단계): 창고 · 장면 종류 · 조건 · 태그 — 공통: 새 주행(회전 공간·가속 제한·후진) + 실제 같은 사람·지게차
mkdir -p ~/agv/claims18; mkdir ~/agv/claims18/$(basename $1)_$2_$4 2>/dev/null || exit 0
cd ~/agv/isaac && source ~/isaacsim-venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
D=$1
if [ "$(basename $D)" = fc ]; then E=$D/fc_env.json; else E=$D/env.json; fi
W=$(cat ~/agv/wm_bev_v5c.best)
AI="--conds wm --wm-ckpt $W --hybrid --pathcand --escape"
case $3 in
  rule) C="--conds respond" ;;
  base) C="--conds none,respond"; case $4 in f2*) C="--conds respond" ;; esac ;;  # 2회차 남은 작업: 대응 없음 빼고 (시간 절약, 사용자 10-10)
  A)    C="$AI" ;;
  B)    C="$AI --hyb-slow 2.5:0.6" ;;
  C)    C="$AI --hyb-slow 2.5:0.6 --tau 0.05" ;;
  F)    C="$AI $(cat ~/agv/final_cfg.txt)" ;;
esac
python logistics.py --v5 --eval5 --backout --swept --smooth --real-motion --types $2 --env $E --out $D --map racks $C --tag _$4 \
    > ~/agv/c18_$(basename $D)_$2_$4.log 2>&1
