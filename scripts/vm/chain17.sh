#!/bin/bash
# v5c: 수집 → 다시 채우기 → 본 학습(v5b 설정, 데이터만 v5c) → 처음 보는 창고 정확도
cd ~/agv && source ~/isaacsim-venv/bin/activate
until [ $(ls ~/agv/claims17 2>/dev/null | grep -vc "^fc_\|^s101_\|^s102_") -ge 36 ] && \
      ! ps -eo cmd | grep "[c]ollect5" | grep -qv "fc_env\|/s101/\|/s102/"; do sleep 30; done
~/isaacsim-venv/bin/python ~/agv/repair17.py > ~/agv/jobs17_repair.txt
xargs -P 3 -L 1 ~/agv/job17.sh < ~/agv/jobs17_repair.txt
while ps -eo cmd | grep -q "[c]ollect5"; do sleep 30; done
mkdir -p ~/agv/v5cdata && cd ~/agv/v5cdata && for d in ~/agv/wh/s00? ~/agv/logistics; do ln -sf $d/wm_data5c/ep*.npz .; done
cd ~/agv
python src/wm_bev.py --train --data ~/agv/v5cdata --out ~/agv/wm_bev_v5c --steps 40000 --cont-scale 2 --term-frac 0.3 --chan-off R \
    --cams 4 --bev-margin 0 --val-max 100 > wm_bev_v5c.log 2>&1
python src/wm_bev.py --eval --data ~/agv/v5cdata --out ~/agv/wm_bev_v5c --show 6 >> wm_bev_v5c.log 2>&1
python -c "import json;L=json.load(open('$HOME/agv/wm_bev_v5c/train_log.json'));print('$HOME/agv/wm_bev_v5c/wm_%d.pt'%max(L,key=lambda r:r['auroc'])['step'])" > wm_bev_v5c.best.tmp \
  || echo FAILED > wm_bev_v5c.best.tmp
mv wm_bev_v5c.best.tmp wm_bev_v5c.best
~/isaacsim-venv/bin/python ~/agv/eval_gen5c.py > ~/agv/eval_gen5c.out 2>&1
echo done > ~/agv/chain17.done
