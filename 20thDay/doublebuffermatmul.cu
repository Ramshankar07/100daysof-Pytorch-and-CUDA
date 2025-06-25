#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define TILE_SIZE 32
__global__ void matrixMulKernel2x2_DoubleBuffered(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    size_t m, size_t n, size_t k) 
{
    const int base_row = blockIdx.y * TILE_SIZE + threadIdx.y * 2;
    const int base_col = blockIdx.x * TILE_SIZE + threadIdx.x * 2;
    
    if (base_row >= m || base_col >= n) return;
    
    // ========== DOUBLE BUFFERING: 2 sets of shared memory ==========
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1]; 
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];  
    
    float c[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    
    const int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;
    
    // ========== PREFETCH FIRST TILE (into buffer 0) ==========
    int load_buffer = 0;
    int compute_buffer = 1;
    
    // first tile 
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int ty = threadIdx.y * 2 + i;
            int tx = threadIdx.x * 2 + j;
            if (ty < TILE_SIZE && tx < TILE_SIZE) {
                int row = blockIdx.y * TILE_SIZE + ty;
                int col = 0 + tx;  
                As[load_buffer][ty][tx] = (row < m && col < k) ? A[row * k + col] : 0.0f;
                
                row = 0 + ty;  
                col = blockIdx.x * TILE_SIZE + tx;
                Bs[load_buffer][ty][tx] = (row < k && col < n) ? B[row * n + col] : 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    // ========== MAIN LOOP WITH DOUBLE BUFFERING ==========
    for (int tile = 0; tile < num_tiles; ++tile) {
        load_buffer = 1 - load_buffer;      // 0→1 or 1→0
        compute_buffer = 1 - compute_buffer; // 1→0 or 0→1
        
        if (tile < num_tiles - 1) {
            const int next_tile_base = (tile + 1) * TILE_SIZE;
            
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int ty = threadIdx.y * 2 + i;
                    int tx = threadIdx.x * 2 + j;
                    if (ty < TILE_SIZE && tx < TILE_SIZE) {
                        int row = blockIdx.y * TILE_SIZE + ty;
                        int col = next_tile_base + tx;
                        As[load_buffer][ty][tx] = (row < m && col < k) ? A[row * k + col] : 0.0f;
                        
                        row = next_tile_base + ty;
                        col = blockIdx.x * TILE_SIZE + tx;
                        Bs[load_buffer][ty][tx] = (row < k && col < n) ? B[row * n + col] : 0.0f;
                    }
                }
            }
        }
        
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            float a[2], b[2];
            a[0] = As[compute_buffer][threadIdx.y * 2][i];
            a[1] = As[compute_buffer][threadIdx.y * 2 + 1][i];
            b[0] = Bs[compute_buffer][i][threadIdx.x * 2];
            b[1] = Bs[compute_buffer][i][threadIdx.x * 2 + 1];
            
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
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (base_row + i < m && base_col + j < n) {
                C[(base_row + i) * n + base_col + j] = c[i][j];
            }
        }
    }
}
// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n) {    

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    matrixMulKernel2x2_DoubleBuffered<<<grid, block>>>(input_a, input_b, output_c, n, n, n);
    
}