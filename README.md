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

**Learned:**  
- Basics of writing a CUDA kernel.
- Understanding of grid, block, and thread hierarchy in CUDA.  
- How to allocate and manage device (GPU) memory using `cudaMalloc`, `cudaMemcpy`, and `cudaFree`.  

### Reading:  
- Read **Chapter 1** of the PMPP book.  
  - Learned about the fundamentals of parallel programming, CUDA architecture, and the GPU execution model.


Enjoy the journey into the depths of neural networks and high-performance computing with PyTorch and CUDA!
