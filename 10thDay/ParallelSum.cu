//C coalesced memory access is important for parallel sum

#include <iostream>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    extern __shared__ float sharedMemory[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    if (i < N) {

        sharedMemory[tid] = A[i] + B[i+blockDim.x];
        _syncthreads();
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int temp = 0;
            if (tid >= stride) {
                temp = sharedMemory[tid - stride];
            }
            __syncthreads();
            sharedMemory[tid] += temp;
            __syncthreads();
        }

        // Write result to global memory
        C[index] = sharedMemory[tid];
    
    }
}

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    float *d_a, *d_b,*d_c;
    cudaMalloc(&d_a,N*sizeof(float));//allocate memory on device
    cudaMalloc(&d_b,N*sizeof(float));
    cudaMalloc(&d_c,N*sizeof(float));
    cudaMemcpy(d_a,A,N*sizeof(float),cudaMemcpyHostToDevice);//copy data from host to device
    cudaMemcpy(d_b,B,N*sizeof(float),cudaMemcpyHostToDevice);
    int blocksize=256;//number of threads in a block
    int gridsize=ceil(N/blocksize);//number of blocks in a grid
    vectorAdd<<<gridsize,blocksize>>>(d_a,d_b,d_c,N);//call the kernel function to perform vector addition
    cudaMemcpy(C,d_c,N*sizeof(float),cudaMemcpyDeviceToHost);//copy data from device to host
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}