#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string>

int main(int argc, char* argv[]) {
    const int N = 512;

    if (argc < 2) {
        std::cerr << "Usage: ./matmul <tile_size>" << std::endl;
        return 1;
    }

    int TILE = std::stoi(argv[1]);

    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 1.0f);
    std::vector<float> C(N * N, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    // for (int i = 0; i < N; i++) {
    //     for (int k = 0; k < N; k++){
    //         for (int j = 0; j < N; j++)  {
    //             C[i * N + j] += A[i * N + k] * B[k * N + j];
    //         }
    //     }
    // }


    for (int ii = 0; ii < N; ii += TILE) {
        for (int kk = 0; kk < N; kk += TILE) {
            for (int jj = 0; jj < N; jj += TILE) {

                int i_end = std::min(ii + TILE, N);
                int k_end = std::min(kk + TILE, N);
                int j_end = std::min(jj + TILE, N);

                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        for (int j = jj; j < j_end; j++) {
                            C[i * N + j] +=
                                A[i * N + k] * B[k * N + j];
                        }
                    }
                }

            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Matrix size: " << N << " x " << N << std::endl;
    std::cout << "Time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "C[0]: " << C[0] << std::endl;

    return 0;
}