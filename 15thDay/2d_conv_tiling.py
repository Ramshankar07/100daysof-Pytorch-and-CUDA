import triton
import torch
import triton.language as tl

@triton.jit
def twod_conv_tiling(Matrix_A, Matrix_B, Matrix_C, sizeX, sizeY, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the current tile
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    # Calculate starting positions for this tile
    row_start = pid_x * BLOCK_SIZE
    col_start = pid_y * BLOCK_SIZE
    
    # Generate row and column indices for this tile
    row_indices = row_start + tl.arange(0, BLOCK_SIZE)
    col_indices = col_start + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for boundary checking
    row_mask = row_indices < sizeY
    col_mask = col_indices < sizeX
    
    # Load input blocks
    a = tl.load(Matrix_A + row_indices[:, None] * sizeX + col_indices[None, :],
                mask=row_mask[:, None] & col_mask[None, :])
    b = tl.load(Matrix_B + row_indices[:, None] * sizeX + col_indices[None, :],
                mask=row_mask[:, None] & col_mask[None, :])
    
    # Perform convolution
    c = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            c[i, j] = a[i, j] * b[i, j]
    
    # Store results
    tl.store(Matrix_C + row_indices[:, None] * sizeX + col_indices[None, :],
             c, mask=row_mask[:, None] & col_mask[None, :])

def test_Matrix():
    sizeX = 8
    sizeY = 8
    BLOCK_SIZE = 2

    # Initialize input matrices
    Matrix_A = torch.randn(sizeY, sizeX, device='cuda', dtype=torch.float32)
    Matrix_B = torch.randn(sizeY, sizeX, device='cuda', dtype=torch.float32)
    Matrix_C = torch.empty_like(Matrix_A)

    # Calculate grid dimensions
    grid = (triton.cdiv(sizeY, BLOCK_SIZE), triton.cdiv(sizeX, BLOCK_SIZE))
    
    # Launch kernel
    twod_conv_tiling[grid](
        Matrix_A, Matrix_B, Matrix_C,
        sizeX, sizeY, BLOCK_SIZE
    )
    
    return Matrix_C

    