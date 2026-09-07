#include <iostream>
#include <vector>
#include <chrono>

int main() {
    const int N = 512;

    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 1.0f);
    std::vector<float> C(N * N, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++){
            for (int j = 0; j < N; j++)  {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
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