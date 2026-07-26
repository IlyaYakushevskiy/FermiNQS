#!/bin/bash
# Architecture assay (learned antisymmetric pair signature) + budget controls.
# Sentinels are appended to a FILE directly: piping a queue script through `tail`
# buffers its stdout, so sentinel-based waiting between scripts silently never fires
# (that bug stalled the previous queue on 2026-07-25).
cd /home/ilya/FermiNQS
MARK=/home/ilya/FermiNQS/logs/arch_queue_progress.log
: > "$MARK"
source .venv/bin/activate
export PYTHONPATH=.
run () {  # run <marker> <args...>
    local mark="$1"; shift
    echo "$(date '+%F %T') START $mark" >> "$MARK"
    python -u tools/pretrain_hf.py "$@" >> "$MARK".$mark 2>&1
    echo "$(date '+%F %T') DONE  $mark rc=$?" >> "$MARK"
}
PS=32
run arch_n6 --N 6 --steps 25000 --pair-sig-hidden $PS
run arch_n3 --N 3 --steps 25000 --pair-sig-hidden $PS
run arch_n5 --N 5 --steps 25000 --pair-sig-hidden $PS --tag _aufbau
run arch_n4 --N 4 --steps 25000 --pair-sig-hidden $PS --tag _m11 --orbitals "0,0;0,1;1,0;1,1"
run long_n4 --N 4 --steps 100000 --tag _aufbau_long
run long_n6 --N 6 --steps 100000 --tag _long
echo "$(date '+%F %T') ALL_DONE" >> "$MARK"
