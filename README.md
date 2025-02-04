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


Enjoy the journey into the depths of neural networks and high-performance computing with PyTorch and CUDA!
