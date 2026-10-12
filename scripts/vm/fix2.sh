#!/bin/bash
# 영상 후보 다시 찍기 (10-11): 창고 경로(fc 또는 wh/s10x) · 장면(ri:seed:danger) · 카메라 X,Y,높이 · 시도 이름
# 새 배치 번호(3~)도 되게 --eval-seeds. 규칙 → Dreamer 같은 장면, 바닥 고정 카메라, GIF 모두 저장
cd ~/agv/isaac && source ~/isaacsim-venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
D=~/agv/$1; E=$D/env.json; [ "$1" = fc ] && E=$D/fc_env.json
s=$(echo $2 | cut -d: -f2)
python logistics.py --v5 --eval5 --backout --swept --smooth --real-motion --only $2 --eval-seeds $s:$((s+1)) --env $E --out $D --map racks \
    --conds respond,wm --wm-ckpt $(cat ~/agv/wm_bev_v5c.best) --hybrid --pathcand --escape --hyb-slow 2.5:0.6 --tau 0.05 \
    --fixed-cam=$3 --gif-all --tag _vf_$4 > ~/agv/vidsearch2/vf_$(basename $1)_$4.log 2>&1
