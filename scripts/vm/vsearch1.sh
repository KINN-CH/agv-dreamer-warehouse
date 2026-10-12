#!/bin/bash
# 영상 후보 찾기 한 작업 (10-11): 창고 경로(fc 또는 wh/s10x) · 새 배치 번호 — 위험 장면만, 규칙이 부딪힌 장면만 Dreamer도 (최종 설정 C)
cd ~/agv/isaac && source ~/isaacsim-venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
D=~/agv/$1; E=$D/env.json; [ "$1" = fc ] && E=$D/fc_env.json
python logistics.py --v5 --eval5 --backout --swept --smooth --real-motion --types danger --eval-seeds $2:$(($2+1)) --rule-first \
    --env $E --out $D --map racks --conds respond,wm --wm-ckpt $(cat ~/agv/wm_bev_v5c.best) \
    --hybrid --pathcand --escape --hyb-slow 2.5:0.6 --tau 0.05 --tag _vs$2 > ~/agv/vidsearch2/$(basename $1)_$2.log 2>&1
