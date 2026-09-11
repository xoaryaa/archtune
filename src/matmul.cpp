#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>
#include <iomanip>


// ============================================================
// Tiled matrix multiplication
//
// TILE is known at COMPILE TIME.
// For example:
//
// matmul_tiled<4>()
// matmul_tiled<8>()
//
// become separately optimized kernels.
// ============================================================

template <int TILE>
void matmul_tiled(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int N
) {
    // Move through matrix block-by-block
    for (int ii = 0; ii < N; ii += TILE) {

        for (int kk = 0; kk < N; kk += TILE) {

            for (int jj = 0; jj < N; jj += TILE) {

                // Handles cases where N is not perfectly
                // divisible by TILE
                int i_end = std::min(ii + TILE, N);
                int k_end = std::min(kk + TILE, N);
                int j_end = std::min(jj + TILE, N);


                // Actual i-k-j matrix multiplication
                // inside the current tile
                for (int i = ii; i < i_end; i++) {

                    for (int k = kk; k < k_end; k++) {

                        for (int j = jj; j < j_end; j++) {

                            C[i * N + j] +=
                                A[i * N + k] *
                                B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}


// ============================================================
// Benchmark one compile-time tile configuration
// ============================================================

template <int TILE>
void benchmark(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int N
) {
    const int WARMUP_RUNS = 5;
    const int TIMED_RUNS = 30;


    // --------------------------------------------------------
    // 1. Warm-up runs
    //
    // We execute the kernel a few times before collecting data.
    // These timings are NOT recorded.
    // --------------------------------------------------------

    for (int run = 0; run < WARMUP_RUNS; run++) {

        // Reset output matrix
        std::fill(C.begin(), C.end(), 0.0f);

        matmul_tiled<TILE>(A, B, C, N);
    }


    // --------------------------------------------------------
    // 2. Timed runs
    // --------------------------------------------------------

    std::vector<double> times;

    times.reserve(TIMED_RUNS);


    for (int run = 0; run < TIMED_RUNS; run++) {

        // Reset C BEFORE starting timer.
        // We do not want matrix-zeroing time included.
        std::fill(C.begin(), C.end(), 0.0f);


        auto start =
            std::chrono::steady_clock::now();


        matmul_tiled<TILE>(A, B, C, N);


        auto end =
            std::chrono::steady_clock::now();


        std::chrono::duration<double, std::milli> elapsed =
            end - start;


        times.push_back(elapsed.count());
    }


    // --------------------------------------------------------
    // 3. Mean
    // --------------------------------------------------------

    double mean =
        std::accumulate(
            times.begin(),
            times.end(),
            0.0
        )
        / times.size();


    // --------------------------------------------------------
    // 4. Sort measurements
    // --------------------------------------------------------

    std::vector<double> sorted = times;

    std::sort(
        sorted.begin(),
        sorted.end()
    );


    // --------------------------------------------------------
    // 5. Median
    // --------------------------------------------------------

    double median;

    int n = static_cast<int>(sorted.size());


    if (n % 2 == 0) {

        median =
            (
                sorted[n / 2 - 1]
                +
                sorted[n / 2]
            )
            / 2.0;

    } else {

        median =
            sorted[n / 2];
    }


    // --------------------------------------------------------
    // 6. Standard deviation
    // --------------------------------------------------------

    double variance = 0.0;


    for (double t : times) {

        double difference =
            t - mean;

        variance +=
            difference * difference;
    }


    variance /=
        times.size();


    double stddev =
        std::sqrt(variance);


    // --------------------------------------------------------
    // 7. Checksum
    //
    // This happens OUTSIDE the timed region.
    //
    // Besides checking correctness, using the result helps make
    // it clear that the calculated matrix is actually needed.
    // --------------------------------------------------------

    double checksum = 0.0;

    for (float value : C) {
        checksum += value;
    }


    // --------------------------------------------------------
    // 8. Print results
    // --------------------------------------------------------

    std::cout
        << std::fixed
        << std::setprecision(4);


    std::cout
        << "Matrix size: "
        << N
        << " x "
        << N
        << "\n";


    std::cout
        << "Tile size: "
        << TILE
        << "\n";


    std::cout
        << "Warmup runs: "
        << WARMUP_RUNS
        << "\n";


    std::cout
        << "Timed runs: "
        << TIMED_RUNS
        << "\n";


    std::cout
        << "Mean: "
        << mean
        << " ms\n";


    std::cout
        << "Median: "
        << median
        << " ms\n";


    std::cout
        << "Min: "
        << sorted.front()
        << " ms\n";


    std::cout
        << "Max: "
        << sorted.back()
        << " ms\n";


    std::cout
        << "Std Dev: "
        << stddev
        << " ms\n";


    std::cout
        << "C[0]: "
        << C[0]
        << "\n";


    std::cout
        << "Checksum: "
        << checksum
        << "\n";
}


// ============================================================
// Main
//
// Main receives a runtime tile-size request.
//
// Example:
//
// ./benchmarks/matmul 4
//
// It then dispatches ONCE into the corresponding compile-time
// specialized benchmark.
// ============================================================

int main(int argc, char* argv[]) {

    const int N = 512;


    // --------------------------------------------------------
    // Check command-line argument
    // --------------------------------------------------------

    if (argc < 2) {

        std::cerr
            << "Usage: ./benchmarks/matmul <tile_size>\n";

        std::cerr
            << "Supported tile sizes: "
            << "4 8 16 32 64 128 256\n";

        return 1;
    }


    // Runtime tile requested by user
    int tile = std::stoi(argv[1]);


    // --------------------------------------------------------
    // Create matrices
    //
    // A = all 1s
    // B = all 1s
    // C = initially 0
    // --------------------------------------------------------

    std::vector<float> A(
        N * N,
        1.0f
    );


    std::vector<float> B(
        N * N,
        1.0f
    );


    std::vector<float> C(
        N * N,
        0.0f
    );


    // --------------------------------------------------------
    // Runtime dispatcher
    //
    // This decision happens ONCE.
    //
    // Once we enter benchmark<4>, for example, TILE=4 is a
    // compile-time constant.
    // --------------------------------------------------------

    switch (tile) {

        case 4:

            benchmark<4>(
                A,
                B,
                C,
                N
            );

            break;


        case 8:

            benchmark<8>(
                A,
                B,
                C,
                N
            );

            break;


        case 16:

            benchmark<16>(
                A,
                B,
                C,
                N
            );

            break;


        case 32:

            benchmark<32>(
                A,
                B,
                C,
                N
            );

            break;


        case 64:

            benchmark<64>(
                A,
                B,
                C,
                N
            );

            break;


        case 128:

            benchmark<128>(
                A,
                B,
                C,
                N
            );

            break;


        case 256:

            benchmark<256>(
                A,
                B,
                C,
                N
            );

            break;
            
        case 512:
            benchmark<512>(A, B, C, N);
            break;

        default:

            std::cerr
                << "Unsupported tile size: "
                << tile
                << "\n";

            std::cerr
                << "Supported tile sizes: "
                << "4 8 16 32 64 128 256\n";

            return 1;
    }


    return 0;
}