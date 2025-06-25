## **Double Buffering Method in CUDA**
Double buffering (ping-pong buffering) **overlaps computation and memory transfers** by using two memory buffers:
1. **One buffer is used for computation**.
2. **The other buffer is being loaded with new data** from global memory.

This technique ensures **the GPU is never idle** and minimizes memory latency.

### **Pros:**
✅ **Better Memory Utilization**: Reduces memory stalls by overlapping data transfer with computation.  
✅ **Improves Performance on Large Matrices**: Beneficial for **large-scale matrix multiplications** where global memory access is a bottleneck.  
✅ **No Precision Issues**: Works with any data type (FP32, FP64, etc.).  

### **Cons:**
❌ **Requires Manual Synchronization**: Needs careful **CUDA stream** and **shared memory** management.  
❌ **Higher Register Usage**: Since multiple buffers must be stored in shared memory.  

### **Performance Factors:**
- **Uses two sets of shared memory buffers** (`A_tile` and `B_tile`).
- **Global memory accesses are coalesced** to avoid unnecessary loads.
- Requires **synchronization (`__syncthreads()`)** to manage buffer swaps.