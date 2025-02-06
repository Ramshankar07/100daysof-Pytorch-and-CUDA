import triton
import torch
import triton.language as tl


@triton.jit
def twod_conv_tiling(Matrix_A,Matrix_B,sizeX,sizeY,BLOCK_SIZE:tl.constexpr):

    pid_x = tl.program_id(0) # we have the rows
    pid_y = tl.program_id(1) # we have the columns
    
    row_start = pid_x*BLOCK_SIZE # calculate the start of the row
    col_start = pid_y*BLOCK_SIZE # calculate the start of the column
    
    row_indices = row_start + tl.arange(0,BLOCK_SIZE) 
    col_indices= col_start+ tl.arrange(0,BLOCK_SIZE)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            
    # get the indices of the rows which is less than BLOCK_SIZE(boundary) to process
    
    
    # tl.store(Matrix_C+flat_indicies,C,mask=valid_mask)


def test_addMatrix():
    sizeX = 8
    sizeY = 8
    Mask=2

    BLOCK_SIZE = 2

    Matrix_A = torch.randn(sizeY, sizeX, device='cuda', dtype=torch.float32)
    Matrix_B = torch.randn(sizeY, sizeX, device='cuda', dtype=torch.float32)

    grid = (triton.cdiv(sizeX, BLOCK_SIZE), triton.cdiv(sizeY, BLOCK_SIZE))
    twod_conv_tiling[grid](Matrix_A, Matrix_B, sizeX, sizeY, BLOCK_SIZE)

    Matrix_C = Matrix_C_flat.reshape(sizeY, sizeX)

    expected = Matrix_A + Matrix_B
    print("Matrix A:\n", Matrix_A)
    print("Matrix B:\n", Matrix_B)
    print("Matrix C (Triton):\n", Matrix_C)
    print("Expected (PyTorch):\n", expected)
    assert torch.allclose(Matrix_C, expected), "Triton result does not match PyTorch result!"

test_addMatrix()

    
    
    