import triton
import torch
import triton.language as tl


@triton.jit
def MatrixTanspose(Matrix_A,Matrix_B,sizeX,sizeY,BLOCK_SIZE:tl.constexpr):

    pid_x = tl.program_id(0) # we have the rows
    pid_y = tl.program_id(1) # we have the columns
    
    row_start = pid_x*BLOCK_SIZE # calculate the start of the row
    col_start = pid_y*BLOCK_SIZE # calculate the start of the column

    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if row_start+i < sizeY and col_start+j < sizeX:
                Matrix_B[col_start+j,row_start+i] = Matrix_A[row_start+i,col_start+j]
    
    tl.store(Matrix_B,Matrix_B)


def __main__():
    sizeX = 8
    sizeY = 8
    BLOCK_SIZE = 256

    Matrix_A = torch.randn(sizeY, sizeX, device='cuda', dtype=torch.float32)
    Matrix_B = torch.zeros_like(Matrix_A, device='cuda', dtype=torch.float32)

    grid = (triton.cdiv(sizeX, BLOCK_SIZE), triton.cdiv(sizeY, BLOCK_SIZE))
    MatrixTanspose[grid](Matrix_A, Matrix_B, sizeX, sizeY, BLOCK_SIZE)



    
    
    