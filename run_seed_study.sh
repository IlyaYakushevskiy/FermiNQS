#!/bin/bash
# Quantify the run-to-run spread of the phase metric. Every number in the thesis is a
# single sample; the same N=4 config at 25k vs 100k gave 0.445 vs 0.528, so differences
# below ~0.1 have not been interpretable. Three seeds at the two decision-relevant points:
# N=6 (the ceiling the whole argument rests on) and N=4 (the intermediate regime).
cd /home/ilya/FermiNQS
MARK=/home/ilya/FermiNQS/logs/seed_study_progress.log
until grep -q MATCHED_DONE /home/ilya/FermiNQS/logs/matched_n4_progress.log 2>/dev/null; do sleep 30; done
: > "$MARK"
source .venv/bin/activate
export PYTHONPATH=.
for SEED in 43 44; do
  echo "$(date '+%F %T') START n6_seed$SEED" >> "$MARK"
  python -u tools/pretrain_hf.py --N 6 --steps 25000 --seed $SEED --tag _s$SEED \
      >> "$MARK".n6_s$SEED 2>&1
  echo "$(date '+%F %T') DONE  n6_seed$SEED rc=$?" >> "$MARK"
done
for SEED in 43 44; do
  echo "$(date '+%F %T') START n4_seed$SEED" >> "$MARK"
  python -u tools/pretrain_hf.py --N 4 --steps 25000 --seed $SEED --tag _m11_s$SEED \
      --orbitals "0,0;0,1;1,0;1,1" >> "$MARK".n4_s$SEED 2>&1
  echo "$(date '+%F %T') DONE  n4_seed$SEED rc=$?" >> "$MARK"
done
echo "$(date '+%F %T') SEED_STUDY_DONE" >> "$MARK"
