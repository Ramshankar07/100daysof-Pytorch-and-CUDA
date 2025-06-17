//C coalesced memory access is important for parallel sum

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void MatrixTranspose(const float* A, const float* B, int N) {
    
   int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( x < N && y < N) {
        B[y * N + x] = A[x * N + y];
    }

    void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << message << " - CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    }
    }
int main() {
    const int H= 10;
    const int W= 10;
    float *A = new float[N * N];
    float *B = new float[N * N];
    for (int i = 0; i < N * N; i++) {
         A[i] = static_cast<float>(rand()) / RAND_MAX;//random values between 0 and 1

    }
    float *d_a, *d_b;
    cudaMalloc(&d_a,N*N*sizeof(float));//allocate memory on device
    cudaMalloc(&d_b,N*N*sizeof(float));
    cudaMemcpy(d_a,A,N*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,B,N*N*sizeof(float),cudaMemcpyHostToDevice);
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    MatrixTranspose<<<blocksize,gridSize>>>(d_a,d_b,N);//call the kernel function to perform vector addition
    cudaDeviceSynchronize();
    cudaMemcpy(B,d_b,N*sizeof(float),cudaMemcpyDeviceToHost);//copy data from device to host
    cudaFree(d_a);
    cudaFree(d_b);
    free(A);
    free(B);

}