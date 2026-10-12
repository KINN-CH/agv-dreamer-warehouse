#!/bin/bash
# v5c 수집 한 작업: 창고 디렉터리 · 첫 ep · 개수 — 길 없는 장면 제외, 실제 같은 사람·지게차, 새 주행(회전 공간·가속 제한·후진)
mkdir -p ~/agv/claims17; mkdir ~/agv/claims17/$(basename $1)_$2 2>/dev/null || exit 0
cd ~/agv/isaac && source ~/isaacsim-venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
D=$1
if [ "$(basename $D)" = logistics ]; then E=$D/simple_env.json; elif [ "$(basename $D)" = fc ]; then E=$D/fc_env.json; else E=$D/env.json; fi
python logistics.py --v5 --collect5 $3 --first $2 --env $E --out $D --map racks --swept --smooth --backout --real-motion --reach \
    --data-dir wm_data5c > ~/agv/c17_$(basename $D)_$2.log 2>&1
