//C coalesced memory access is important for parallel sum

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void LayerNorm(const float* A, const float* B, int N) {
    
    int tid = threadIdx.x;
    float *sum = &row[blockDim.x];
    int i = blockIdx.x * blockDim.x + tid;
    if (i < N) {
        extern __shared__ float sharedMem[];
        float *row = sharedMem;
        for (int col = threadIdx.y; col < N; col += blockDim.y) {
            row_data[col] = A[row * N + col];
        }
        _syncthreads();
        float mean = 0.0f;
        for (int col = 0; col < N; col++) {
            mean += row_data[col];
        }
        mean /= N;

        // Compute variance
        float variance = 0.0f;
        for (int col = 0; col < N; col++) {
            variance += (row_data[col] - mean) * (row_data[col] - mean);
        }
        variance /= N;
        float stddev = sqrtf(variance + 1e-7);

        // Normalize
        for (int col = threadIdx.y; col < N; col += blockDim.y) {
            B[row * N + col] = (row_data[col] - mean) / stddev;
        }
    }
}
int main() {
    const int N = 10;

    float *A = new float[N * N];
    float *B = new float[N * N];
    for (int i = 0; i < N * N; i++) {
         A[i] = static_cast<float>(rand()) / RAND_MAX;//random values between 0 and 1

    }
    float *d_a, *d_b;
    cudaMalloc(&d_a,N*N*sizeof(float));//allocate memory on device
    cudaMalloc(&d_b,N*N*sizeof(float));
    cudaMemcpy(d_a,A,N*N*sizeof(float),cudaMemcpyHostToDevice);
    int blocksize=256;//number of threads in a block
    size_t sharedMemory = (N+2*blocksize)*sizeof(float)//number of blocks in a grid
    LayerNorm<<<N,blocksize,sharedMemory>>>(d_a,d_b,N);//call the kernel function to perform vector addition
    cudaMemcpy(B,d_b,N*sizeof(float),cudaMemcpyDeviceToHost);//copy data from device to host
    cudaFree(d_a);
    cudaFree(d_b);

}