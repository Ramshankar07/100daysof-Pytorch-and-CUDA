#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <crand.h>
// This code won't have backpropagation and grads, infernence : That's tomorrow's work
#define CheckCudaError(err) { 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

}

#define CheckCUDNNError(err) { 
    cudnnStatus_t err = (func);
    if(err != CUDNN_STATUS_SUCCESS) {
        std::cerr<<"CUDNN Error: " << cudnnGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

const int batch_size = 8;
const int input_size = 600;
const int hidden_size = 1000;
const int output_size = 10;

const float learning_rate = 0.01;
const int num_epochs = 4;


void init_weights(float *weights, int size) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, weights, size);
    curandDestroyGenerator(gen);
}
void init_bias(float *bias, int size) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    curandGenerateUniform(gen, bias, size);
    curandDestroyGenerator(gen);
}
int main(){

    cudnnHandle_t handle;
    CheckCUDNNError(cudnnCreate(&handle));

    cudnnTensorDescriptor_t input_desc, hidden_desc, hidden_desc2, output_desc;
    CheckCUDNNError(cudnnCreateTensorDescriptor(&input_desc));
    CheckCUDNNError(cudnnCreateTensorDescriptor(&hidden_desc));
    CheckCUDNNError(cudnnCreateTensorDescriptor(&hidden_desc2));
    CheckCUDNNError(cudnnCreateTensorDescriptor(&output_desc));

    CheckCUDNNError(cudnnSetTensor4dDescriptor(input_desc,CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, input_size, 1, 1));
    CheckCUDNNError(cudnnSetTensor4dDescriptor(hidden_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, hidden_size, 1, 1));
    CheckCUDNNError(cudnnSetTensor4dDescriptor(hidden_desc2, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, hidden_size, 1, 1));
    CheckCUDNNError(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, output_size, 1, 1));
    
    
    cudnnActivationDescriptor_t relu_desc;
    CheckCUDNNError(cudnnCreateActivationDescriptor(&relu_desc));
    CheckCUDNNError(cudnnSetActivationDescriptor(relu_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0f));
    
    cudnnFilterDescriptor_t weight_desc, weight_desc2, weight_desc3;
    CheckCUDNNError(cudnnCreateFilterDescriptor(&weight_desc));
    CheckCUDNNError(cudnnCreateFilterDescriptor(&weight_desc2));
    CheckCUDNNError(cudnnCreateFilterDescriptor(&weight_desc3));
    CheckCUDNNError(cudnnSetFilter4dDescriptor(weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, hidden_size, input_size, 1, 1));
    CheckCUDNNError(cudnnSetFilter4dDescriptor(weight_desc2, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, hidden_size, hidden_size, 1, 1));
    CheckCUDNNError(cudnnSetFilter4dDescriptor(weight_desc3, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, output_size, hidden_size, 1, 1));
    
    cudnnConvolutionDescriptor_t conv_desc, conv_desc2;
    CheckCUDNNError(cudnnCreateConvolutionDescriptor(&conv_desc));
    CheckCUDNNError(cudnnCreateConvolutionDescriptor(&conv_desc2));
    CheckCUDNNError(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CheckCUDNNError(cudnnSetConvolution2dDescriptor(conv_desc2, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    //mem alloc for NN weights and bias
    float *weights, *weights2, *weights3;
    float *bias, *bias2, *bias3;
    size_t weights_size = hidden_size * input_size * sizeof(float);
    size_t weights2_size = hidden_size * hidden_size * sizeof(float);
    size_t weights3_size = output_size * hidden_size * sizeof(float);
    size_t bias_size = hidden_size * sizeof(float);
    size_t bias2_size = hidden_size * sizeof(float);
    size_t bias3_size = output_size * sizeof(float);
    CheckCudaError(cudaMalloc(&weights, weights_size));
    CheckCudaError(cudaMalloc(&weights2, weights2_size));
    CheckCudaError(cudaMalloc(&weights3, weights3_size));
    CheckCudaError(cudaMalloc(&bias, bias_size));
    CheckCudaError(cudaMalloc(&bias2, bias2_size));
    CheckCudaError(cudaMalloc(&bias3, bias3_size));
    CheckCudaError(cudaMemset(weights, 0, weights_size));
    CheckCudaError(cudaMemset(weights2, 0, weights2_size));
    CheckCudaError(cudaMemset(weights3, 0, weights3_size));
    CheckCudaError(cudaMemset(bias, 0, bias_size));
    CheckCudaError(cudaMemset(bias2, 0, bias2_size));
    CheckCudaError(cudaMemset(bias3, 0, bias3_size));
    //mem alloc for NN input, hidden, hidden2, output
    float *input, *hidden, *hidden2, *output;
    size_t input_size = batch_size * input_size * sizeof(float);
    size_t hidden_size = batch_size * hidden_size * sizeof(float);
    size_t hidden2_size = batch_size * hidden_size * sizeof(float);
    size_t output_size = batch_size * output_size * sizeof(float);
    
    CheckCudaError(cudaMalloc(&input, batch_size*input_size*sizeof(float)));
    CheckCudaError(cudaMalloc(&hidden, batch_size*hidden_size*sizeof(float)));
    CheckCudaError(cudaMalloc(&hidden2, batch_size*hidden2_size*sizeof(float)));
    CheckCudaError(cudaMalloc(&output, batch_size*output_size*sizeof(float)));
    
    //with dummy data
    init_weights(weights, weights_size);
    init_weights(weights2, weights2_size);
    init_weights(weights3, weights3_size);
    init_bias(bias, bias_size);
    init_bias(bias2, bias2_size);
    init_bias(bias3, bias3_size);
    init_weights(input, input_size);
    init_weights(output, output_size);

    //training loop
    for(int i=0; i<num_epochs;i++){

        float alpha= 1.0f; float beta= 0.0f;
        //first layer
        CheckCUDNNError(cudnnConvolutionForward(handle, &alpha, input_desc, input, weight_desc, weights, conv_desc, conv_desc, &beta, hidden_desc, hidden));
        CheckCUDNNError(cudnnActivationForward(handle, relu_desc, &alpha, hidden_desc, hidden, &beta, hidden_desc2, hidden2));
        //second layer
        CheckCUDNNError(cudnnConvolutionForward(handle, &alpha, hidden_desc2, hidden2, weight_desc2, weights2, conv_desc2, conv_desc2, &beta, hidden_desc, hidden));
        CheckCUDNNError(cudnnActivationForward(handle, relu_desc, &alpha, hidden_desc, hidden, &beta, hidden_desc2, hidden2));
        //third layer
        CheckCUDNNError(cudnnConvolutionForward(handle, &alpha, hidden_desc2, hidden2, weight_desc3, weights3, conv_desc2, conv_desc2, &beta, output_desc, output));
        CheckCUDNNError(cudnnActivationForward(handle, relu_desc, &alpha, hidden_desc, hidden, &beta, hidden_desc2, hidden2));

       
        Loss=0.0f;
        for(int j=0; j<batch_size; j++){
            Loss+=fabs(output[j*output_size] - output[j*output_size]);

            }
        CheckCudaError(cudaMemcpy(input, output, batch_size*output_size*sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "Epoch: " << i << " Loss: " << Loss << std::endl;
            
    }
    CheckCudaError(cudaFree(weights));
    CheckCudaError(cudaFree(weights2));
    CheckCudaError(cudaFree(weights3));
    CheckCudaError(cudaFree(bias));
    CheckCudaError(cudaFree(bias2));
    CheckCudaError(cudaFree(bias3));
    CheckCudaError(cudaFree(input));
    CheckCudaError(cudaFree(hidden));
    CheckCudaError(cudaFree(hidden2));
    CheckCudaError(cudaFree(output));
    CheckCUDNNError(cudnnDestroy(handle));
    CheckCUDNNError(cudnnDestroyTensorDescriptor(input_desc));
    CheckCUDNNError(cudnnDestroyTensorDescriptor(hidden_desc));
    CheckCUDNNError(cudnnDestroyTensorDescriptor(hidden_desc2));
    CheckCUDNNError(cudnnDestroyTensorDescriptor(output_desc));
    CheckCUDNNError(cudnnDestroyActivationDescriptor(relu_desc));
    CheckCUDNNError(cudnnDestroyFilterDescriptor(weight_desc));
    CheckCUDNNError(cudnnDestroyFilterDescriptor(weight_desc2));
    CheckCUDNNError(cudnnDestroyFilterDescriptor(weight_desc3));
    CheckCUDNNError(cudnnDestroyConvolutionDescriptor(conv_desc));
    CheckCUDNNError(cudnnDestroyConvolutionDescriptor(conv_desc2));
    return 0;




}   


