# 100 Days of PyTorch and CUDA

Welcome to my journey through "100 Days of PyTorch and CUDA"! This repository is dedicated to improving my abilities and maintaining consistency in learning and optimizing with PyTorch and CUDA over the next 100 days. Each day, I'll tackle new concepts, implement algorithms, and share insights and progress.

## About the Challenge

The challenge is to spend each day learning or coding with PyTorch and CUDA, with the aim of deepening my understanding of these powerful tools for machine learning and parallel computing. The goal is not just to learn but to apply this knowledge in practical, real-world scenarios.

# Progress 

Day 1: Introduction to PyTorch
Learned the basics of tensors in PyTorch and how they differ from NumPy arrays. Implemented simple tensor operations and explored PyTorch's dynamic computation graph.

Day 2: Experimenting with tensors and autograd
Day 3: Experimenting with types of activation functions
Day 4: Experimenting with optimizers
Day 5: CNN model for MNIST dataset

# CUDA 
## Day 6: Vector Addition with CUDA
### File: `vectorAdd.cu`
**Summary:**  
Implemented vector addition by writing a simple CUDA program. Explored how to launch a kernel to perform a parallelized addition of two arrays, where each thread computes the sum of a pair of values.  

## Day 7
### File: `triton-vectadd.py`
**Summary:**  
Worked on vect addition using triton.

## Day 8
### File: `MatrixAdd.cu`,`Matrixadd.py`
**Summary:**  
Implemented Matrix addition in both triton and CUDA.

## Day 9
### File: `Matrix_vec_mult.cu`
**Summary:**  
Implemented matrix-vector multiplication using CUDA. Each thread was set up to compute the dot product between a matrix row and the given vector. Optimized performance using shared memory.  

## Day 10
### File: `PartialSum.cu`
**Summary:**  
Worked on parallel reduction to compute the partial sum of an array. Implemented a tree-based reduction algorithm, minimizing warp divergence for better performance.  

## Day 11
### File: `LayerNorm.cu`
**Summary:**  
Implemented Layer Normalization in CUDA, often used in deep learning models. Explored normalization techniques across batches and layers using reduction operations. Addressed the challenge of maintaining numerical stability during computation.  

## Day 12
### File: `MatrixTranspose.cu`
**Summary:**  
Implemented CUDA-based matrix transposition. Optimized the implementation by leveraging shared memory to minimize global memory reads and writes. Ensured proper handling of edge cases when the matrix dimensions are not multiples of the block size.  

## Day 13

### File: `one_d_convolution.cu`
**Summary:**  
Implemented a simple 1D convolution algorithm using CUDA. This involved sliding a kernel (or filter) over an input array and computing the weighted sum of elements. Each thread was assigned to compute the convolution at a specific position in the output array.  


## Day 14
### File: `2d_convolution_with_tiling.cu`  
**Summary:**  
Implemented a 2D convolution algorithm with tiling optimization using CUDA. Divided the input matrix into tiles and leveraged shared memory to minimize global memory accesses, ensuring efficient computation of the convolution kernel across the matrix. Handled boundary conditions using halo cells to process edges and corners correctly.  

## Day 15
### File: `2d_conv_tiling.py`
Same thing in triton

## Day 16
### File: `sparse-matrix.cu`
This CUDA kernel implements a hybrid ELL-COO sparse matrix format for efficient sparse matrix-vector multiplication (SpMV) on GPUs

## Day 17
### File: `Cuda_CNN.ipynb`
This CUDA kernel implements a whole CNN architecture with pooling layers 

## Day 18
### File: `1dconv.py`
**Summary:**  
Implemented 1D convolution using Triton, a high-level language for GPU programming. The implementation uses block-based processing where each thread block handles a segment of the input array. Features include proper boundary handling with padding, efficient memory access patterns, and support for variable kernel sizes. The convolution operation computes weighted sums by sliding the kernel over the input data.

## Day 19
### File: `tiled-matrix.cu`
**Summary:**  
Implemented tiled matrix multiplication in CUDA with advanced optimization techniques. Each thread computes a 2x2 block of the output matrix, utilizing shared memory tiles to minimize global memory accesses. The implementation includes loop unrolling for better instruction-level parallelism and proper boundary condition handling. This approach significantly improves memory bandwidth utilization and reduces memory latency.

## Day 20
### File: `doublebuffermatmul.cu`
**Summary:**  
Implemented double-buffered matrix multiplication using CUDA, a technique that overlaps computation with memory transfers. The kernel uses two sets of shared memory buffers (ping-pong buffering) where one buffer is used for computation while the other is being loaded with new data from global memory. This approach ensures the GPU is never idle and minimizes memory latency, making it particularly effective for large-scale matrix multiplications where global memory access is a bottleneck.

##Day 21
### file: `relu.cu`
**Summary:**
Implemented Tiling based relu activation for a 2D matrix

## Day 22
### File: `cuda-NN.cu`
**Summary:**
Implemented a complete neural network in CUDA with forward and backward passes, including matrix operations, activation functions, and gradient computation.

---

# Advanced CUDA Learning Roadmap (Days 23-100)

Starting from Day 23, I'll be following a comprehensive advanced CUDA learning roadmap covering:

## 🚀 **Phase 1: Memory and Access Patterns (Days 23-35)**
- Matrix Transpose optimization
- Array Reversal with coalesced access
- 1D Stencil computations (Heat Equation)

## 🔄 **Phase 2: Reduction Patterns (Days 36-48)**
- Sum Reduction with warp primitives
- Maximum Element with Index
- Histogram Computation with atomics

## 🤝 **Phase 3: Synchronization and Cooperation (Days 49-61)**
- Parallel Prefix Sum (Scan algorithms)
- Stream Compaction
- Cooperative Groups for BFS and Quicksort

## 🧮 **Phase 4: Computational Patterns (Days 62-80)**
- Advanced Matrix Multiplication (GEMM)
- Triangular Matrix Solve (TRSM)
- 2D Convolution with FFT

## ⚡ **Phase 5: Advanced Optimizations (Days 81-88)**
- Kernel Fusion (LSTM, Softmax, LayerNorm)
- Irregular Parallelism (SpMV, KNN, Ray Tracing)

## 🔧 **Phase 6: System-Level Patterns (Days 89-95)**
- Multi-Stream Programming
- Dynamic Parallelism
- Pipeline Parallel Training

## 🎯 **Phase 7: Domain-Specific Challenges (Days 96-100)**
- Flash Attention implementation
- Fused Adam Optimizer
- Quantized GEMM with Tensor Cores

**📋 Detailed roadmap available in [CUDA_ROADMAP.md](CUDA_ROADMAP.md)**

---

# Setup and Usage

## Quick Start
```bash
# Windows (PowerShell)
.\cuda_build.ps1 compile
.\cuda_build.ps1 run-all

# Linux/macOS (Bash)
./cuda_build.sh compile
./cuda_build.sh run-all

# Cross-platform (Make)
make all
make run-all
```

**📖 Complete setup guide available in [CUDA_SETUP.md](CUDA_SETUP.md)**

---

Enjoy the journey into the depths of neural networks and high-performance computing with PyTorch and CUDA!
