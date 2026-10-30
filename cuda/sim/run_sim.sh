#!/usr/bin/env bash
#
# Run matmul_gpu on GPGPU-Sim (the execution-driven simulator inside
# Accel-Sim). Meant to be run INSIDE the accelsim/ubuntu-18.04_cuda-11
# container, with the repo's cuda/ directory mounted at /work:
#
#   docker run --platform linux/amd64 --rm -v "$PWD":/work -v accelsim:/accel-sim \
#       accelsim/ubuntu-18.04_cuda-11 bash /work/sim/run_sim.sh 256 smem32
#
#   ./run_sim.sh [N] [kernel ...]
#
# Each kernel gets its own simulator process and log:
#   sim/out/<config>/N<N>/<kernel>.log
# Then sim/parse_sim.py turns the logs into a table.

set -euo pipefail

N=${1:-256}
shift || true
KERNELS=${@:-naive_uncoalesced naive smem8 smem16 smem32 smem_rt8 smem_rt16 smem_rt32 regtile}

# Simulated GPU. SM7_QV100 = Tesla V100 (Volta, 80 SMs), the most
# heavily validated config in Accel-Sim.
CONFIG=${CONFIG:-SM7_QV100}

ACCEL=/accel-sim
# The image presets CUDA_INSTALL_PATH=/usr/local/cuda-11, which doesn't exist
export CUDA_INSTALL_PATH=/usr/local/cuda
export PATH=$CUDA_INSTALL_PATH/bin:$PATH

# ------------------------------------------------------------
# 1. Build the simulator once (cached in the "accelsim" volume)
# ------------------------------------------------------------

# We only need the execution-driven simulator (GPGPU-Sim), not
# Accel-Sim's trace-driven front end, which needs traces captured on
# a real GPU. Pinned to the v4.2.1 release: the current dev branch
# needs a newer compiler than this image's gcc 7.5.
GPGPUSIM_TAG=v4.2.1
GPGPUSIM=$ACCEL/gpgpu-sim-$GPGPUSIM_TAG

if [ ! -d "$GPGPUSIM" ]; then
    git clone --depth 1 -b "$GPGPUSIM_TAG" https://github.com/gpgpu-sim/gpgpu-sim_distribution.git "$GPGPUSIM"
fi

# The simulator's environment script isn't written for set -eu
set +eu
source "$GPGPUSIM/setup_environment" release
set -eu

if ! ls "$GPGPUSIM"/lib/gcc-*/cuda-*/release/libcudart.so >/dev/null 2>&1; then
    make -C "$GPGPUSIM" -j"$(nproc)"
    set +eu
    source "$GPGPUSIM/setup_environment" release
    set -eu
fi

# ------------------------------------------------------------
# 2. Build the benchmark against the simulator
#
# -cudart shared: so the simulator's libcudart.so is loaded
#                 instead of the real CUDA runtime
# compute_70 PTX: GPGPU-Sim executes PTX, so it must be embedded
# ------------------------------------------------------------

BIN=/work/sim/matmul_gpu_sim
nvcc -O3 -std=c++14 -DARCHTUNE_SIM \
    -gencode arch=compute_70,code=compute_70 \
    -cudart shared \
    -o "$BIN" /work/matmul_gpu.cu

# ------------------------------------------------------------
# 3. Run each kernel in its own directory with the GPU config
# ------------------------------------------------------------

OUT=/work/sim/out/$CONFIG/N$N
mkdir -p "$OUT"

for k in $KERNELS; do
    RUN=$(mktemp -d)
    cp "$GPGPUSIM/configs/tested-cfgs/$CONFIG/"* "$RUN/"
    echo "== $k (N=$N, $CONFIG) =="
    start=$(date +%s)
    (cd "$RUN" && "$BIN" "$N" "$k") > "$OUT/$k.log" 2>&1 || echo "  (non-zero exit, see $OUT/$k.log)"
    end=$(date +%s)
    echo "wall_seconds = $((end - start))" >> "$OUT/$k.log"
    echo "  done in $((end - start)) s"
    rm -rf "$RUN"
done
