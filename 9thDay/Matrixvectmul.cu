#include <iostream>
#include <cuda_runtime.h>

__device__ float randomFunction(float x, float y)
{
    return x * y;
}

__global__ void matrixFunction(const float *A, const float *B, float *C, const int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        C[i + size * j] = randomFunction(A[i + size * j], B[ j]);
    }
}

int main()
{   
    int N = 8;
    int BLOCK_SIZE = 64;
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(N + BLOCK_SIZE - 1 / BLOCK_SIZE, N + BLOCK_SIZE - 1 / BLOCK_SIZE);
    int size = sizeof(float) * N * N;

    float *A,*B,*C;
    float *dA,*dB,*dC;
    A = new float[N*N];
    B = new float[N]; //since we are doing matrix vector multiplication
    C = new float[N];
    size_t sizeB = sizeof(float) * N;
    cudaMalloc((void**)&dA,size);
    cudaMalloc((void**)&dB,sizeB);
    cudaMalloc((void**)&dC,size);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i + N * j] = 1.0f; 
            
        }
        B[i] = 2.0f;
        C[i] = 0.0f;
    }
    
    cudaMemcpy(dA,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,size,cudaMemcpyHostToDevice);

    // now we have everything set up
    matrixFunction<<<gridDim,blockDim>>>(dA,dB,dC,N);
    cudaDeviceSynchronize();

    cudaMemcpy(C,dC,size,cudaMemcpyDeviceToHost);

    for (int i = 0; i < N*N; i++) {
        std::cout << C[i] << " ";
        if ((i + 1) % N == 0) std::cout << std::endl;
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}