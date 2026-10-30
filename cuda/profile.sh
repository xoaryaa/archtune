#!/usr/bin/env bash
#
# Profile each kernel with Nsight Compute.
#
#   ./profile.sh [N] [kernel ...]
#
# For each kernel this writes:
#   reports/<kernel>.ncu-rep   full report (open in the Nsight Compute GUI)
#   reports/<kernel>.csv       the handful of metrics the README discusses
#
# Only one launch per kernel is profiled, after 2 warmup launches,
# and only our own kernels (matmul_*) are matched, not cuBLAS.

set -euo pipefail

N=${1:-4096}
shift || true
KERNELS=${@:-naive_uncoalesced naive smem8 smem16 smem32 smem_rt8 smem_rt16 smem_rt32 regtile}

METRICS="gpu__time_duration.sum"
METRICS+=",sm__throughput.avg.pct_of_peak_sustained_elapsed"                   # compute busy %
METRICS+=",gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"   # memory busy %
METRICS+=",dram__bytes_read.sum"                                               # global traffic
METRICS+=",sm__warps_active.avg.pct_of_peak_sustained_active"                  # achieved occupancy
METRICS+=",launch__registers_per_thread"
METRICS+=",smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"      # coalescing efficiency
METRICS+=",l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"           # shared-mem bank conflicts
METRICS+=",smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"   # stalled on __syncthreads
METRICS+=",smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"  # stalled on global loads

mkdir -p reports

for k in $KERNELS; do
    echo "== $k =="
    ncu --set full --import-source yes \
        -k regex:^matmul_ --launch-skip 2 --launch-count 1 \
        -f -o "reports/$k" \
        ./matmul_gpu "$N" "$k"

    ncu -i "reports/$k.ncu-rep" --csv --page raw --metrics "$METRICS" > "reports/$k.csv"
done
