#include<iostream>
#include<cuda_runtime.h>

using namespace std;

__global__ void vectorAdd(const float *a, const float *b, float *c, const int N) {
    // int i = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("blockIdx.x: %d\n", blockIdx.x);
    // printf("blockDim.x: %d\n", blockDim.x);
    // printf("i: %d\n", i);
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 3;
    size_t nBytes = N * sizeof(float);

    // allocate vectors in host memory
    float *h_a = (float *) malloc(nBytes);
    float *h_b = (float *) malloc(nBytes);
    float *h_c = (float *) malloc(nBytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // allocate vectors in device array
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, nBytes);
    cudaMalloc((void **) &d_b, nBytes);
    cudaMalloc((void **) &d_c, nBytes);

    // copy memory from host memory to device memory
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, nBytes, cudaMemcpyHostToDevice);

    // launch kernel 
    int numberofblocks = 1;
    int numberofthreads = N;
    vectorAdd<<<numberofblocks, numberofthreads>>>(d_a, d_b, d_c, N);

    // copy memory from device memory to host memory
    cudaMemcpy(h_c, d_c, nBytes, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for (int i = 0; i < N; i++) {
        cout << h_c[i] << endl;
    }

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
    
}