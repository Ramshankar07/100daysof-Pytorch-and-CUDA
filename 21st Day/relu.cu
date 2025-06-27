#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 128
#define TILE_SIZE 32
__global__ void relu(const float* __restrict__ A, float* __restrict__ C,size_t m, size_t n) {
    
    const int base_row = blockIdx.y * TILE_SIZE + threadIdx.y ;
    const int base_col= blockIdx.x*TILE_SIZE+ threadIdx.x;
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1]; 

    if (base_row < m && base_col < n) {
        int idx = base_row*n + base_col;
        
        As[threadIdx.y][threadIdx.x]=A[idx];
    }
    else {
        As[threadIdx.y][threadIdx.x]=0.0f;
    }
    __syncthreads();
    if (base_row < m && base_col < n){
        
        int idx = base_row*n + base_col;
        C[idx]=fmaxf(0.0f, As[threadIdx.y][threadIdx.x]);
    }    
    }

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {    
    dim3 block(TILE_SIZE, TILE_SIZE); 
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,(m + TILE_SIZE - 1) / TILE_SIZE);
    relu<<<grid, block>>>(input, output, m, n);
    
}