# Archtune

Archtune is a hardware-aware performance optimization project for studying how implementation choices interact with the underlying hardware.

Current setup: **C++ on Apple M4**.

---

## what I've done so far

I started with a basic 512×512 matrix multiplication with:

- different loop orders
- cache tiling
- static vs runtime tile sizes
- 30 runs after 5 warmup runs for every configuration


The main thing I've learned so far is that performance is a lot less predictable than just saying "better cache locality = faster."

---

## static tile size

For this version, the tile size is hard-coded in the C++ file and the program is recompiled for each value.

| Tile Size | Mean (ms) | Median (ms) | Std Dev |
|---:|---:|---:|---:|
| 8   | 5.062 | 5.086 | 0.121 |
| 16  | 8.698 | 8.758 | 0.333 |
| 32  | 6.728 | 7.150 | 1.173 |
| 64  | 10.467 | 10.480 | 0.477 |
| 128 | 11.211 | 11.186 | 0.707 |

**Tile size 8 was the fastest here.**

The latency didn't follow a clean trend as the tile size increased. It was expected that the result to mostly come down to cache behavior.

---

## with dynamic tile size

Instead of changing the source and recompiling every time, I changed the program so I could run:

```bash
./matmul 256 
```
Now the tile size is only known at runtime.

The results changed quite a bit — in this version, tile size 256 gave the lowest latency.

So the same tiling idea produced a different "best" tile size depending on whether the value was known at compile time or runtime.

With a hard-coded tile size, the compiler already knows the loop bounds while generating the executable. That can give it more room for things like constant propagation, loop optimization, and vectorization.

With a runtime value, it has to generate code that works for whatever tile size the user gives it.

So this isn't really "compile time getting added to runtime." It's more that the generated machine code itself can be different.

That was probably the most useful thing from these experiments.

## GPU version (CUDA)

`cuda/matmul_gpu.cu` runs the same experiment on an NVIDIA GPU, with N = 4096 by default (512 is too small to keep a GPU busy). It uses the same protocol: 5 warmup runs and 30 timed runs, timed with CUDA events. Every result is checked against cuBLAS, and cuBLAS itself is spot-checked against a CPU double-precision dot product.

| kernel | idea |
|---|---|
| `naive_uncoalesced` | one thread per output, but `threadIdx.x` walks rows, so warp loads are strided |
| `naive` | the only change is that `threadIdx.x` walks columns, which makes loads coalesced |
| `smem8/16/32` | shared-memory tiling with the tile size as a **template parameter** (compile time) |
| `smem_rt8/16/32` | the same kernel with the tile size taken from `blockDim` (**runtime**) |
| `regtile` | 128×128 block tile, with each thread computing 8×8 outputs held in registers |
| `cublas` | vendor baseline |

The `smem` vs `smem_rt` pair is the GPU version of the static vs dynamic tile experiment above.

```bash
cd cuda
make                      # or: make ARCH=sm_75 for a T4
./matmul_gpu              # all kernels, N=4096
./matmul_gpu 2048 smem32  # one kernel
./profile.sh              # Nsight Compute report + metrics CSV per kernel -> reports/
```

### simulated results (GPGPU-Sim)

I don't have an NVIDIA GPU (Apple M4), so I first ran the kernels on **GPGPU-Sim v4.2.1**, a cycle-level NVIDIA GPU simulator. It ran in the Accel-Sim Docker image under x86 emulation, using the **RTX 2060 config** (Turing, 30 SMs, 1365 MHz). Every kernel's output was checked against a CPU reference.

```bash
cd cuda
docker run --platform linux/amd64 --rm -e CONFIG=SM75_RTX2060 \
    -v "$PWD":/work -v accelsim:/accel-sim accelsim/ubuntu-18.04_cuda-11 \
    bash /work/sim/run_sim.sh 256 naive smem16 smem32 smem_rt16 smem_rt32 regtile
python3 sim/parse_sim.py sim/out/SM75_RTX2060/N256
```

N = 256. The six kernels took about 31 minutes of simulation in total.

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
- **Compile-time vs runtime tile size shows up on the GPU too.** The runtime versions take 12% (tile 16) and 8% (tile 32) more cycles. They also have *higher* IPC, because they execute more instructions: index math and loop overhead that the template versions unroll away. So higher IPC doesn't mean faster. It's the same lesson as the CPU experiment.
- **`regtile` is starved for parallelism at this size.** 128×128 block tiles over a 256×256 matrix give only 4 blocks for 30 SMs, so most of the GPU is idle. The kernel is built for large N. It also has 262K shared-memory bank conflicts on its `As` reads, which is the next thing to fix.
- Earlier, a V100 run at N=128 showed the coalescing effect: `naive_uncoalesced` took **4.0× more cycles** than `naive` (142K vs 36K).

These numbers come from a simulator model, not real hardware. They're best read as relative comparisons between kernels.



