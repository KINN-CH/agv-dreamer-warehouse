#!/bin/bash
# 긴 영상용 주행 (10-11): 창고 경로(fc 또는 wh/s10x) · 장면(ri:seed:danger) · 따라가는 카메라 높이(천장보다 낮게) · 이름
# 규칙 → Dreamer 같은 장면, 출발~도착 전체를 위에서 따라가며 + 깊이·2D 지도·상상·길찾기 지도 기록 (make_panel_1012.py로 합성)
cd ~/agv/isaac && source ~/isaacsim-venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
D=~/agv/$1; E=$D/env.json; [ "$1" = fc ] && E=$D/fc_env.json
s=$(echo $2 | cut -d: -f2)
python logistics.py --v5 --eval5 --backout --swept --smooth --real-motion --only $2 --eval-seeds $s:$((s+1)) --env $E --out $D --map racks \
    --conds respond,wm --wm-ckpt $(cat ~/agv/wm_bev_v5c.best) --hybrid --pathcand --escape --hyb-slow 2.5:0.6 --tau 0.05 \
    --follow-cam $3 --dump-vis --log-look --tag _panel_$4 > ~/agv/vidsearch2/panel_$4.log 2>&1
