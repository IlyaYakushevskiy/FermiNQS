#!/bin/bash
# Settle whether the N=4 pair-signature gain is real or just extra optimisation.
# The 25k comparison is confounded: the baseline itself improves ~16% from 25k->100k
# (0.707 -> 0.591 on the (0,2) member), which is the same size as the pair-sig "gain".
# This runs BOTH arms on the SAME member (1,1) at the SAME budget (100k).
cd /home/ilya/FermiNQS
MARK=/home/ilya/FermiNQS/logs/matched_n4_progress.log
until grep -q ALL_DONE /home/ilya/FermiNQS/logs/arch_queue_progress.log 2>/dev/null; do sleep 30; done
: > "$MARK"
source .venv/bin/activate
export PYTHONPATH=.
ORB="0,0;0,1;1,0;1,1"
echo "$(date '+%F %T') START baseline_m11_100k" >> "$MARK"
python -u tools/pretrain_hf.py --N 4 --steps 100000 --tag _m11_long --orbitals "$ORB" \
    >> "$MARK".baseline 2>&1
echo "$(date '+%F %T') DONE  baseline_m11_100k rc=$?" >> "$MARK"
echo "$(date '+%F %T') START pairsig_m11_100k" >> "$MARK"
python -u tools/pretrain_hf.py --N 4 --steps 100000 --tag _m11_long --orbitals "$ORB" \
    --pair-sig-hidden 32 >> "$MARK".pairsig 2>&1
echo "$(date '+%F %T') DONE  pairsig_m11_100k rc=$?" >> "$MARK"
echo "$(date '+%F %T') MATCHED_DONE" >> "$MARK"
