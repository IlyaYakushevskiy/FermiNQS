#!/bin/bash
cd /home/ilya/FermiNQS
MARK=/home/ilya/FermiNQS/logs/n78_progress.log
: > "$MARK"
source .venv/bin/activate
export PYTHONPATH=.
for N in 7 8; do
  echo "$(date '+%F %T') START n$N" >> "$MARK"
  python -u tools/pretrain_hf.py --N $N --steps 25000 --tag _aufbau >> "$MARK".n$N 2>&1
  echo "$(date '+%F %T') DONE  n$N rc=$?" >> "$MARK"
done
echo "$(date '+%F %T') N78_DONE" >> "$MARK"
