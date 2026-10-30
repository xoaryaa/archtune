#include <cuda_runtime.h>
#ifndef ARCHTUNE_SIM
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>


// ============================================================
// GPU matrix multiplication: C = A * B (row-major, N x N, fp32)
//
// Same question as src/matmul.cpp, but on an NVIDIA GPU:
// how do implementation choices (memory access pattern, tiling,
// compile-time vs runtime tile size, register blocking) change
// what the hardware actually does?
//
// Kernels, roughly in order of sophistication:
//
//   naive_uncoalesced  1 thread / output, threadIdx.x walks ROWS
//   naive              1 thread / output, threadIdx.x walks COLS
//   smem<TILE>         shared-memory tiling, TILE known at compile time
//   smem_rt            shared-memory tiling, tile only known at runtime
//   regtile            2D block tiling + 8x8 outputs per thread in registers
//   cublas             vendor baseline (also the correctness reference)
//
// Building with -DARCHTUNE_SIM targets the GPGPU-Sim / Accel-Sim
// simulator instead of a real GPU: cuBLAS is left out (the
// simulator cannot run it), the reference is computed on the
// CPU, and each kernel is launched once.
// ============================================================


#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                         cudaGetErrorString(err_), __FILE__, __LINE__);    \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

#ifndef ARCHTUNE_SIM
#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t st_ = (call);                                       \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",             \
                         static_cast<int>(st_), __FILE__, __LINE__);       \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)
#endif


// ============================================================
// 1. Naive, UNcoalesced
//
// Consecutive threads in a warp (threadIdx.x) get consecutive
// ROWS. So for A[row * N + k], the 32 threads of a warp read
// addresses N floats apart -> 32 separate memory transactions
// instead of one. The C store has the same problem.
//
// This is the GPU version of the "bad loop order" experiment.
// ============================================================

__global__ void matmul_naive_uncoalesced(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; k++) {
            acc += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}


// ============================================================
// 2. Naive, coalesced
//
// Only change: threadIdx.x now maps to COLUMNS.
// A warp reads 32 consecutive floats of B (one 128-byte line)
// and all threads in the warp read the same A element
// (broadcast). Same math, very different memory traffic.
// ============================================================

__global__ void matmul_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; k++) {
            acc += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}


// ============================================================
// 3. Shared-memory tiling, TILE known at COMPILE TIME
//
// Each block loads a TILE x TILE piece of A and B into shared
// memory once, then every thread reuses it TILE times. This
// cuts global memory reads by a factor of TILE.
//
// Because TILE is a template parameter, the inner k-loop has a
// constant trip count and can be fully unrolled.
// ============================================================

template <int TILE>
__global__ void matmul_smem(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    for (int t = 0; t < N; t += TILE) {

        // Out-of-range elements are loaded as 0 so N does not
        // have to be a multiple of TILE
        As[ty][tx] = (row < N && t + tx < N) ? A[row * N + t + tx] : 0.0f;
        Bs[ty][tx] = (t + ty < N && col < N) ? B[(t + ty) * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}


// ============================================================
// 4. Shared-memory tiling, tile only known at RUNTIME
//
// Identical algorithm to matmul_smem<TILE>, but the tile size
// comes from blockDim and shared memory is sized at launch.
//
// This mirrors the static vs dynamic tile experiment on the
// CPU: same idea, but the compiler can no longer unroll the
// inner loop or fold the index math into constants.
// ============================================================

__global__ void matmul_smem_rt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    extern __shared__ float smem[];

    const int tile = blockDim.x;
    float* As = smem;
    float* Bs = smem + tile * tile;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * tile + ty;
    int col = blockIdx.x * tile + tx;

    float acc = 0.0f;

    for (int t = 0; t < N; t += tile) {

        As[ty * tile + tx] = (row < N && t + tx < N) ? A[row * N + t + tx] : 0.0f;
        Bs[ty * tile + tx] = (t + ty < N && col < N) ? B[(t + ty) * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < tile; k++) {
            acc += As[ty * tile + k] * Bs[k * tile + tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}


// ============================================================
// 5. 2D block tiling + register tiling
//
// A block computes a BM x BN tile of C. Each of its
// (BM*BN)/(TM*TN) threads computes a TM x TN sub-tile and keeps
// those outputs in registers.
//
// Per k-step, a thread loads TM values of A and TN values of B
// from shared memory into registers and does TM*TN FMAs with
// them. That raises arithmetic intensity per shared-memory load
// from 1 FMA/load (smem kernel) to TM*TN/(TM+TN) = 4 FMA/load.
//
// Requires N to be a multiple of BM and BN (no bounds checks).
// ============================================================

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
matmul_regtile(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Which TM x TN piece of the block tile this thread owns
    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    // Move pointers to the start of this block's row/column
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // How threads cooperate to copy A (BM x BK) and B (BK x BN)
    // tiles into shared memory
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    constexpr int STRIDE_A = NUM_THREADS / BK;

    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    constexpr int STRIDE_B = NUM_THREADS / BN;

    float acc[TM * TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    for (int bk = 0; bk < N; bk += BK) {

        #pragma unroll
        for (int off = 0; off < BM; off += STRIDE_A) {
            As[(innerRowA + off) * BK + innerColA] =
                A[(innerRowA + off) * N + innerColA];
        }

        #pragma unroll
        for (int off = 0; off < BK; off += STRIDE_B) {
            Bs[(innerRowB + off) * BN + innerColB] =
                B[(innerRowB + off) * N + innerColB];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        #pragma unroll
        for (int k = 0; k < BK; k++) {

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                regA[i] = As[(threadRow * TM + i) * BK + k];
            }

            #pragma unroll
            for (int j = 0; j < TN; j++) {
                regB[j] = Bs[k * BN + threadCol * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i * TN + j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            C[(threadRow * TM + i) * N + threadCol * TN + j] = acc[i * TN + j];
        }
    }
}

constexpr int RT_BM = 128;
constexpr int RT_BN = 128;
constexpr int RT_BK = 8;
constexpr int RT_TM = 8;
constexpr int RT_TN = 8;


// ============================================================
// cuBLAS
//
// cuBLAS is column-major. A row-major C = A * B is the same
// memory as a column-major C^T = B^T * A^T, so we pass B first.
// ============================================================

#ifndef ARCHTUNE_SIM
void matmul_cublas(cublasHandle_t handle, const float* A, const float* B, float* C, int N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N, &alpha, B, N, A, N, &beta, C, N));
}
#endif


// ============================================================
// Timing
//
// Same protocol as the CPU benchmark: 5 warmup runs, 30 timed
// runs. Timed with CUDA events, which measure GPU execution
// time on the stream (not launch overhead on the host).
//
// C is zeroed before every run. All kernels write into the same
// buffer, so without this a kernel that silently wrote nothing
// would "pass" using the previous kernel's correct output.
// ============================================================

struct Stats {
    double mean;
    double median;
    double min;
    double max;
    double stddev;
};

Stats time_kernel(const std::function<void()>& launch, float* dC, size_t bytes) {
#ifdef ARCHTUNE_SIM
    // The simulator is deterministic and every launch costs
    // minutes of wall time, so one run per kernel is enough.
    // Event times here are NOT meaningful; read cycles from the
    // simulator log instead (see sim/parse_sim.py).
    const int WARMUP_RUNS = 0;
    const int TIMED_RUNS = 1;
#else
    const int WARMUP_RUNS = 5;
    const int TIMED_RUNS = 30;
#endif

    for (int run = 0; run < WARMUP_RUNS; run++) {
        CUDA_CHECK(cudaMemset(dC, 0, bytes));
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> times;
    times.reserve(TIMED_RUNS);

    for (int run = 0; run < TIMED_RUNS; run++) {
        CUDA_CHECK(cudaMemset(dC, 0, bytes));

        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    Stats s;
    s.mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    int n = static_cast<int>(sorted.size());
    s.median = (n % 2 == 0) ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 : sorted[n / 2];
    s.min = sorted.front();
    s.max = sorted.back();

    double variance = 0.0;
    for (double t : times) {
        variance += (t - s.mean) * (t - s.mean);
    }
    s.stddev = std::sqrt(variance / times.size());

    return s;
}


// ============================================================
// Correctness
//
// Compare against the cuBLAS result. Inputs are random in
// [-1, 1] (all-ones inputs hide indexing bugs, since every
// element of C would be N no matter which A/B elements a thread
// read).
// ============================================================

bool check(const std::vector<float>& got, const std::vector<float>& ref, double& max_err) {
    max_err = 0.0;
    bool ok = true;
    for (size_t i = 0; i < ref.size(); i++) {
        double err = std::fabs(static_cast<double>(got[i]) - ref[i]);
        max_err = std::max(max_err, err);
        if (err > 1e-2 + 1e-3 * std::fabs(ref[i])) {
            ok = false;
        }
    }
    return ok;
}


// ============================================================
// Main
//
// ./matmul_gpu [N] [kernel]
//
// N defaults to 4096 (512 is too small to fill a modern GPU).
// kernel defaults to "all". Passing a single kernel name is
// what profile.sh uses so Nsight Compute only sees that kernel.
// ============================================================

struct KernelEntry {
    std::string name;
    std::function<void()> launch;
    bool supported;
};

int main(int argc, char* argv[]) {

    int N = (argc > 1) ? std::stoi(argv[1]) : 4096;
    std::string only = (argc > 2) ? argv[2] : "all";

    // --------------------------------------------------------
    // Device info
    // --------------------------------------------------------

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

#ifndef ARCHTUNE_SIM
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    int mem_clock_khz = 0;
    int bus_width_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, dev));

    // DDR: 2 transfers per clock
    double peak_bw_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8) / 1e9;

    std::cout << "Device: " << prop.name
              << " (sm_" << prop.major << prop.minor
              << ", " << prop.multiProcessorCount << " SMs"
              << ", ~" << std::fixed << std::setprecision(0) << peak_bw_gbs << " GB/s peak DRAM)\n";
#else
    std::cout << "Device: GPGPU-Sim (see gpgpusim.config)\n";
#endif
    std::cout << "Matrix size: " << N << " x " << N << "\n\n";

    // --------------------------------------------------------
    // Data
    // --------------------------------------------------------

    const size_t count = static_cast<size_t>(N) * N;
    const size_t bytes = count * sizeof(float);

    std::vector<float> hA(count), hB(count), hC(count), hRef(count);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

#ifdef ARCHTUNE_SIM
    // --------------------------------------------------------
    // Reference result computed fully on the CPU (i-k-j order,
    // double accumulation). Fine for the small N we simulate.
    // --------------------------------------------------------

    {
        std::vector<double> acc(count, 0.0);
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                double a = hA[i * N + k];
                for (int j = 0; j < N; j++) {
                    acc[i * N + j] += a * hB[k * N + j];
                }
            }
        }
        for (size_t i = 0; i < count; i++) {
            hRef[i] = static_cast<float>(acc[i]);
        }
    }
#else
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // --------------------------------------------------------
    // Reference result from cuBLAS, spot-checked on the CPU
    // in double precision so we are not trusting cuBLAS blindly
    // --------------------------------------------------------

    matmul_cublas(handle, dA, dB, dC, N);
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC, bytes, cudaMemcpyDeviceToHost));

    std::uniform_int_distribution<int> idx(0, N - 1);
    for (int s = 0; s < 64; s++) {
        int i = idx(rng);
        int j = idx(rng);
        double expect = 0.0;
        for (int k = 0; k < N; k++) {
            expect += static_cast<double>(hA[i * N + k]) * hB[k * N + j];
        }
        if (std::fabs(expect - hRef[i * N + j]) > 1e-2 + 1e-3 * std::fabs(expect)) {
            std::cerr << "cuBLAS reference disagrees with CPU at (" << i << ", " << j << ")\n";
            return 1;
        }
    }
#endif

    // --------------------------------------------------------
    // Kernel table
    // --------------------------------------------------------

    auto grid_for = [N](int tile) {
        return dim3((N + tile - 1) / tile, (N + tile - 1) / tile);
    };

    auto smem_rt = [&](int tile) {
        return [&, tile]() {
            size_t shmem = 2 * tile * tile * sizeof(float);
            matmul_smem_rt<<<grid_for(tile), dim3(tile, tile), shmem>>>(dA, dB, dC, N);
        };
    };

    bool regtile_ok = (N % RT_BM == 0) && (N % RT_BN == 0);

    std::vector<KernelEntry> kernels = {
#ifndef ARCHTUNE_SIM
        {"cublas", [&]() { matmul_cublas(handle, dA, dB, dC, N); }, true},
#endif

        {"naive_uncoalesced", [&]() {
            matmul_naive_uncoalesced<<<grid_for(32), dim3(32, 32)>>>(dA, dB, dC, N);
        }, true},

        {"naive", [&]() {
            matmul_naive<<<grid_for(32), dim3(32, 32)>>>(dA, dB, dC, N);
        }, true},

        {"smem8",  [&]() { matmul_smem<8><<<grid_for(8), dim3(8, 8)>>>(dA, dB, dC, N); }, true},
        {"smem16", [&]() { matmul_smem<16><<<grid_for(16), dim3(16, 16)>>>(dA, dB, dC, N); }, true},
        {"smem32", [&]() { matmul_smem<32><<<grid_for(32), dim3(32, 32)>>>(dA, dB, dC, N); }, true},

        {"smem_rt8",  smem_rt(8),  true},
        {"smem_rt16", smem_rt(16), true},
        {"smem_rt32", smem_rt(32), true},

        {"regtile", [&]() {
            dim3 grid(N / RT_BN, N / RT_BM);
            dim3 block((RT_BM * RT_BN) / (RT_TM * RT_TN));
            matmul_regtile<RT_BM, RT_BN, RT_BK, RT_TM, RT_TN><<<grid, block>>>(dA, dB, dC, N);
        }, regtile_ok},
    };

    // --------------------------------------------------------
    // Run
    // --------------------------------------------------------

    const double flops = 2.0 * N * N * static_cast<double>(N);
    double cublas_median = 0.0;
    bool any = false;
    bool all_ok = true;

    std::cout << std::left << std::setw(20) << "kernel"
              << std::right
              << std::setw(12) << "median ms"
              << std::setw(12) << "mean ms"
              << std::setw(10) << "std ms"
              << std::setw(11) << "GFLOP/s"
              << std::setw(11) << "% cuBLAS"
              << std::setw(12) << "max err"
              << "  check\n";

    for (const auto& k : kernels) {
        if (only != "all" && only != k.name) continue;
        any = true;

        if (!k.supported) {
            std::cout << std::left << std::setw(20) << k.name
                      << "skipped (N must be a multiple of " << RT_BM << ")\n";
            continue;
        }

        Stats s = time_kernel(k.launch, dC, bytes);

        CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        bool ok = check(hC, hRef, max_err);
        all_ok = all_ok && ok;

        if (k.name == "cublas") cublas_median = s.median;
        double gflops = flops / (s.median * 1e-3) / 1e9;

        std::cout << std::left << std::setw(20) << k.name
                  << std::right << std::fixed
                  << std::setprecision(3) << std::setw(12) << s.median
                  << std::setprecision(3) << std::setw(12) << s.mean
                  << std::setprecision(3) << std::setw(10) << s.stddev
                  << std::setprecision(0) << std::setw(11) << gflops;

        if (cublas_median > 0.0) {
            std::cout << std::setprecision(1) << std::setw(10) << 100.0 * cublas_median / s.median << "%";
        } else {
            std::cout << std::setw(11) << "-";
        }

        std::cout << std::scientific << std::setprecision(2) << std::setw(12) << max_err
                  << "  " << (ok ? "ok" : "FAIL") << "\n";
    }

    if (!any) {
        std::cerr << "Unknown kernel: " << only << "\nAvailable: all";
        for (const auto& k : kernels) std::cerr << " " << k.name;
        std::cerr << "\n";
        return 1;
    }

#ifndef ARCHTUNE_SIM
    CUBLAS_CHECK(cublasDestroy(handle));
#endif
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return all_ok ? 0 : 1;
}
