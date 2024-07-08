#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void printtt(){

    printf(" threadIdx.X %d , blockIdx.X: %d, BlockIdx.Y: %d, GridDim.x: %d, GridDim.Y: %d",threadIdx.x,blockIdx.x,blockIdx.y,gridDim.x,gridDim.y);

}

int main(){
    int nx,ny,nz;
    ny=8;
    nx=8;
    nz=8;

    dim3 block(2,2,2);
    dim3 grid(nx/block.x,ny/block.y,nz/block.z);
    printtt << <grid,block>> >();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}