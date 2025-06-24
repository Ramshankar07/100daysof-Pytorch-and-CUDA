#include<stdio.h>
#include<cuda_runtime.h>

#define TILE_SIZE 32
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void tiledMatrixMul(const float* __restrict__ A,const float* __restrict__ B, float* __restrict__ C, size_t M, size_t N, size_t K) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    if (row >= M || col >= N) return;

    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];
    // Initializing shared memory for the tiles which uses 2x2 blocks
    float c[2][2]= {{0.0f, 0.0f}, {0.0f, 0.0f}};
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        const int tile = t*TILE_SIZE;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int rowIndex = threadIdx.y* 2 + i;
                int colIndex = threadIdx.x * 2 + j;
                if (rowIndex < TILE_SIZE && colIndex < TILE_SIZE) {
                    if (tileSize + i * TILE_SIZE + j < K) {
                    int r = blockIdx.y * TILE_SIZE + ty;
                    int c = tile + tx;
                    As[ty][tx] = (r < M && c < K) ? A[r * K + c] : 0.0f;
                    r = tile + ty;
                    c= blockIdx.x * TILE_SIZE + tx;
                    Bs[ty][tx] = (r < K && c < N) ? B[r * N + c] : 0.0f;
                    }
                }
                
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            // Each thread computes a 2x2 block of the output matrix
            float a[2],b[2];
            a[0] = As[ty*2][i];
            a[1] = As[ty*2 + 1][i];
            b[0] = Bs[i][tx*2];
            b[1] = Bs[i][tx*2 + 1];
            #pragma unroll
            for (int y = 0; y < 2; ++y) {
                #pragma unroll
                for (int x = 0; x < 2; ++x) {
                    c[y][x] += a[y] * b[x];
                }

            }
        }
        __syncthreads();

    }
    //output
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (row + i < M && col + j < N) {
                C[(row + i) * N + (col + j)] = c[i][j];
            }
        }
    }

}


extern "C" void solution(const float* input_a, const float* input_b, float* output_c,
                        size_t m, size_t n, size_t k) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, 
              (m + TILE_SIZE - 1) / TILE_SIZE);
    
    tiledMatrixMul<<<grid, block>>>(input_a, input_b, output_c, m, n, k);
    cudaDeviceSynchronize();
}