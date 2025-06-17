import triton
import triton.language as tl

@triton.jit
def conv1d( a_ptr, b_ptr, c_ptr,n_elements,kernel_size: tl.constexpr ,  BLOCK_SIZE: tl.constexpr ):
    pid = tl.program_id(axis=0) 
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    
    padding_left = (kernel_size - 1) // 2
    mask_out = offsets < n_elements
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Main convolution loop
    for i in range(kernel_size):
        b_i = tl.load(b_ptr + i, cache_modifier=".cg")
        # Load input segment with boundary checks
        a_offsets = offsets - padding_left +i
        mask = (a_offsets >= 0) & (a_offsets < n_elements)
        a = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
        
        # Multiply-accumulate
        acc += a * b_i
    
    # Store results
    tl.store(c_ptr + offsets, acc, mask=mask_out)


# Note: A, B, C are all float32 device tensors
def solution(A, B, C, N: int, K: int):
    BLOCK_SIZE = 128
    grid = lambda _: (triton.cdiv(N, BLOCK_SIZE),)
    conv1d[grid](A, B, C, N, K, BLOCK_SIZE)
    return C