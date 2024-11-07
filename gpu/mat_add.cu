#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixAdd(const float *a, const float *b, float *c, const int rows, const int cols) {
    // map thread and block index to matrix coordinate (x, y)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // map matrix coordinate to linear global memory index (offset)
    int index = i * cols + j;

    if (i < rows && j < cols) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n_rows = 5;
    int n_cols = 10;
    size_t nBytes = n_rows * n_cols * sizeof(float);
    int dimx = 3;
    int dimy = 3;
    dim3 block(dimx, dimy);
    // integer division truncates towards 0
    dim3 grid((n_rows + block.x - 1) / block.x, (n_cols + block.y - 1) / block.y);

    // initialize vectors in host memory
    float *h_a = (float *) malloc(nBytes);
    float *h_b = (float *) malloc(nBytes);
    float *h_c = (float *) malloc(nBytes);

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            int index = i * n_cols + j;
            h_a[index] = 1;
            h_b[index] = 2;
        }
    }

    // initialize vectors in device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, nBytes);
    cudaMalloc((void **) &d_b, nBytes);
    cudaMalloc((void **) &d_c, nBytes);

    // copy memory from host to device 
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, nBytes, cudaMemcpyHostToDevice);

    // launch kernel
    matrixAdd<<<grid, block>>>(d_a, d_b, d_c, n_rows, n_cols);

    // copy memory from device to host
    cudaMemcpy(h_c, d_c, nBytes, cudaMemcpyDeviceToHost);

    // free cuda memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            int index = i * n_cols + j;
            cout << h_c[index] << " ";
        }
        cout << endl;
    }

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}