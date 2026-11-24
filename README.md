# ArchTune

ArchTune is a study of how implementation choices in matrix multiplication interact with the hardware underneath them: cache tiling on a CPU, then nine CUDA GEMM kernels run on a real NVIDIA GPU and on a cycle-level GPU simulator.

The main thing I've learned is that performance is much less predictable than "better locality = faster", and that a simulator and real silicon can disagree about how much an optimization is worth.

## Key results

- **Register tiling reaches 73% of cuBLAS on an NVIDIA T4** (2.88 TFLOP/s at N=4096, 82% of cuBLAS at N=2048). It is 3.6× faster than the best shared-memory kernel.
- **The simulator ranked that same kernel as the slowest tiled kernel.** At the small size I could afford to simulate (N=256), it had only 4 blocks for 30 SMs. The poor result came from grid starvation, not from the kernel design.
- **Simulator and hardware disagree on magnitudes.** On the T4, uncoalesced loads cost 7.5× (4.0× simulated) and runtime tile sizes cost 16–37% (8–12% simulated), while shared-memory tiling gained only 1.7× (3.2× simulated).
- **Higher IPC doesn't mean faster.** In simulation, the runtime-tile kernels had higher IPC than the compile-time ones and were still slower, because they execute extra index math and loop overhead.

## Repository layout

```text
src/matmul.cpp            CPU tiled matmul (512×512, fp32)
cuda/matmul_gpu.cu        9 CUDA GEMM kernels + cuBLAS baseline and correctness checks
cuda/Makefile             build (ARCH=native, or e.g. sm_75 for a T4)
cuda/profile.sh           Nsight Compute report + metrics CSV per kernel
cuda/sim/run_sim.sh       run kernels on GPGPU-Sim inside the Accel-Sim container
cuda/sim/parse_sim.py     turn simulator logs into a table
cuda/sim/out/             raw simulator logs (RTX 2060 and V100 configs)
```

---

## 1. CPU: tiled matmul on Apple M4

A 512×512 fp32 matrix multiply with cache tiling (i-k-j order inside each tile), timed as 30 runs after 5 warmup runs per configuration.

### Tile size hard-coded in the source

For this version, the tile size was hard-coded and the program recompiled for each value.

| Tile size | Mean (ms) | Median (ms) | Std dev (ms) |
|---:|---:|---:|---:|
| 8   | 5.062 | 5.086 | 0.121 |
| 16  | 8.698 | 8.758 | 0.333 |
| 32  | 6.728 | 7.150 | 1.173 |
| 64  | 10.467 | 10.480 | 0.477 |
| 128 | 11.211 | 11.186 | 0.707 |

Tile size 8 was fastest at 5.06 ms (≈53 GFLOP/s). The latency did not follow a clean trend as the tile size grew, even though I expected it to come down mostly to cache behavior.

### Tile size chosen on the command line

I then changed the program to take the tile size as an argument (`./benchmarks/matmul 256`). In that version, tile size 256 gave the lowest latency, so the "best" tile size changed between the two versions.

**Caveat on the cause.** In the current `src/matmul.cpp`, the command-line value goes through a `switch` that calls `benchmark<TILE>()`, so the inner loops are still compiled separately for each tile size, with the tile as a compile-time constant. That means the two versions do not differ in compile-time vs runtime specialization, and that can't be what explains the different best tile. The cause is still open. It could be differences between the two builds, or measurement noise (tile 32 has a 1.17 ms standard deviation). To actually test runtime specialization, the next step is a version where the tile is a plain `int` parameter, built and timed under identical conditions.

### Build and run

```bash
clang++ -O3 -std=c++17 src/matmul.cpp -o benchmarks/matmul   # use the same flags for every comparison
./benchmarks/matmul 8        # supported: 4 8 16 32 64 128 256
```

---

## 2. CUDA: nine GEMM kernels

`cuda/matmul_gpu.cu` implements the kernels below. On a real GPU, every result is checked against cuBLAS, and cuBLAS itself is spot-checked against a CPU double-precision dot product. Inputs are random in [-1, 1].

| kernel | idea |
|---|---|
| `naive_uncoalesced` | one thread per output, but `threadIdx.x` walks rows, so warp loads are strided |
| `naive` | the only change: `threadIdx.x` walks columns, so loads are coalesced |
| `smem8/16/32` | shared-memory tiling, tile size as a **template parameter** (compile time) |
| `smem_rt8/16/32` | the same kernel with the tile size taken from `blockDim` (**runtime**) |
| `regtile` | 128×128 block tile, BK=8, each thread computes 8×8 outputs held in registers |
| `cublas` | vendor baseline |

The `smem` vs `smem_rt` pair is a direct test of compile-time vs runtime tile size on the GPU, where the template version really does fix the loop bounds at compile time.

```bash
cd cuda
make ARCH=sm_75              # T4; or sm_80 (A100), sm_89 (L4/L40S), sm_90 (H100); default: native
./matmul_gpu                 # all kernels, N=4096
./matmul_gpu 2048 smem32     # one kernel at one size
./profile.sh 2048 smem32     # Nsight Compute report + metrics CSV -> reports/
```

---

## 3. Measured on an NVIDIA T4

**Setup:** Tesla T4 (Turing, sm_75, 40 SMs, ~320 GB/s DRAM, 8.1 TFLOP/s FP32 peak) on Google Colab, built with `make ARCH=sm_75`. Median of 30 timed runs after 5 warmups, timed with CUDA events. Every kernel's output matched cuBLAS.

| kernel | N=4096 (ms) | GFLOP/s | % cuBLAS | % cuBLAS at N=2048 |
|---|---:|---:|---:|---:|
| cublas | 34.8 | 3,945 | 100.0% | 100.0% |
| regtile | 47.7 | 2,880 | **73.0%** | **82.1%** |
| smem32 | 173.0 | 795 | 20.1% | 22.4% |
| smem_rt32 | 237.7 | 578 | 14.7% | 16.5% |
| smem16 | 279.7 | 491 | 12.5% | 13.8% |
| naive | 297.5 | 462 | 11.7% | 14.5% |
| smem_rt16 | 354.1 | 388 | 9.8% | 10.8% |
| smem8 | 482.1 | 285 | 7.2% | 8.6% |
| smem_rt8 | 559.1 | 246 | 6.2% | 7.0% |
| naive_uncoalesced | 2,226.7 | 62 | 1.6% | 1.8% |

What this shows:

- **Register tiling is the step that matters.** At N=4096 there are 1,024 blocks of 128×128, enough to fill all 40 SMs. `regtile` is 3.6× faster than `smem32`. cuBLAS reaches about 49% of FP32 peak and `regtile` about 36%.
- **Coalescing costs 7.5×.** Changing only which index `threadIdx.x` walks takes the naive kernel from 2,227 ms to 297 ms.
- **Runtime tile sizes cost 16–37%**, and the cost grows with tile size (tile 8: 16%, 16: 27%, 32: 37%).
- **Small shared-memory tiles barely help, or hurt.** `smem16` is only 6% faster than `naive`, and `smem8` is slower than `naive`. One hypothesis, not yet checked: the T4's caches already capture much of the coalesced naive kernel's reuse, and small tiles do too little work per `__syncthreads()`. Profiling `naive` would test this.
- **Numerics.** At N=4096 every kernel matched cuBLAS exactly. At N=2048 all of them differ from cuBLAS by the same 1.83e-4. The likely explanation is that all my kernels sum along k in the same order, while cuBLAS picks a different algorithm at that size.

### Nsight Compute: smem32 at N=2048

| metric | value |
|---|---:|
| registers / thread | 42 |
| achieved occupancy | ~100% |
| shared-memory bank conflicts | 0 |
| SM throughput | 75% of peak |
| memory throughput | 75% of peak |

Occupancy and bank conflicts are not what limits `smem32`. The profile is consistent with a heavy load of shared-memory instructions, since each FMA needs two shared-memory reads. That is exactly what register tiling reduces, and `regtile` is the next row up in the table.

Nsight locks GPU clocks while profiling, so its kernel duration (42 ms) is longer than the event-timed median (22.4 ms). Use the event timings for speed and Nsight for the reasons behind it.

---

## 4. Simulated with GPGPU-Sim

Before I had access to an NVIDIA GPU, I ran the kernels on **GPGPU-Sim v4.2.1**, a cycle-level NVIDIA GPU simulator, in the Accel-Sim Docker image under x86 emulation. The main config was the **RTX 2060** (Turing, 30 SMs, 1365 MHz), with an earlier V100 run. Every kernel's output was checked against a CPU reference.

```bash
cd cuda
docker run --platform linux/amd64 --rm -e CONFIG=SM75_RTX2060 \
    -v "$PWD":/work -v accelsim:/accel-sim accelsim/ubuntu-18.04_cuda-11 \
    bash /work/sim/run_sim.sh 256 naive smem16 smem32 smem_rt16 smem_rt32 regtile
python3 sim/parse_sim.py sim/out/SM75_RTX2060/N256
```

At N=256, the six kernels took about 31 minutes of simulation in total.

| kernel | cycles | GFLOP/s | IPC | occupancy % | L2 accesses | smem bank conflicts |
|---|---:|---:|---:|---:|---:|---:|
| naive | 285,417 | 160 | 346.5 | 99.9 | 137,984 | 0 |
| smem16 | 97,810 | 468 | 959.5 | 98.8 | 227,840 | 0 |
| smem32 | 89,020 | 515 | 824.5 | 99.9 | 139,264 | 0 |
| smem_rt16 | 109,751 | 417 | 1183.5 | 96.8 | 228,672 | 0 |
| smem_rt32 | 96,078 | 477 | 1128.2 | 99.9 | 139,264 | 0 |
| regtile | 269,158 | 170 | 83.2 | 25.0 | 98,304 | 262,144 |

What this shows:

- **Shared-memory tiling: 3.2× fewer cycles** than naive (`smem32`).
- **Runtime tile sizes take 8–12% more cycles** (tile 32: 8%, tile 16: 12%), despite *higher* IPC. They execute more instructions, the index math and loop overhead that the template versions unroll away.
- **`regtile` is starved at this size.** 128×128 block tiles over a 256×256 matrix give only 4 blocks for 30 SMs, hence 25% occupancy. It also shows 262K shared-memory bank conflicts on its A-tile reads.
- A V100 run at N=128 showed the coalescing effect: `naive_uncoalesced` took **4.0× more cycles** than `naive` (142K vs 36K).

The simulator preloads copied data into L2, and at these sizes the matrices fit in L2, so L2 accesses are the useful traffic measure here, not DRAM traffic.

---

## 5. Simulator vs hardware

| effect | GPGPU-Sim | T4 (N=4096) |
|---|---|---|
| uncoalesced vs coalesced naive | 4.0× more cycles (V100, N=128) | **7.5× slower** |
| smem32 vs naive | 3.2× fewer cycles | **1.7× faster** |
| runtime vs compile-time tile size | 8–12% more cycles | **16–37% slower** |
| regtile | slowest tiled kernel (grid-starved) | **fastest kernel, 73% of cuBLAS** |

These are comparisons of trends, not matched experiments. The simulated runs used small matrices (N=256 and N=128) and different GPU models (RTX 2060, V100), because cycle-level simulation is slow. The direction of each effect held on real hardware, but the sizes did not. The simulator overstated the benefit of shared-memory tiling, understated the costs of uncoalesced access and runtime tile sizes, and could not show `regtile`'s real performance at a size it could afford to run.

---

## 6. Next steps

- **regtile:** fix the A-tile bank conflicts (e.g. by padding the shared-memory tile), then re-measure on the T4 and profile with Nsight.
- **naive vs small tiles:** profile `naive` and `smem16` and compare L1/L2 hit rates, to test why small tiles barely help.
- **CPU:** write a true runtime-tile version (the tile as a plain `int`), build both versions with identical flags, and compare their timings and assembly.
- **Closing the gap to cuBLAS:** vectorized `float4` loads and double buffering, then a tensor-core (WMMA, FP16/BF16) version.
- **Matched comparison:** run the simulator and the T4 at the same N, to compare simulator and hardware like for like.
