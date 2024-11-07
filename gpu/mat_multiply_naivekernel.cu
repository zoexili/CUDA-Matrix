#include<iostream>
#include<cuda_runtime.h>
#include<chrono>

using namespace std;

// naive kernel 
#define BLOCK_SIZE 32
// error checking macro
#define CHECK(call)                                               \
{                                                                 \
    const cudaError_t error = call;                               \
    if (error != cudaSuccess) {                                   \
        printf("Error: %s:%d, ", __FILE__, __LINE__);             \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                  \
    }                                                             \
}

struct Matrix {
    int height;
    int width;
    float *elements;
};

__global__ void matrixMultiply(const Matrix a, const Matrix b, Matrix c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float cvalue = 0.0f;

    if (i < a.height && j < b.width) {
        for (int k = 0; k < a.width; k++) {
            // c[i][j] = sum_k a[i][k] * b[k][j]
            int index_i = i * a.width + k;
            int index_j = k * b.width + j;
            cvalue += a.elements[index_i] * b.elements[index_j];
        }
        int index_c = i * c.width + j;
        c.elements[index_c] = cvalue;
    }
}

void initializeMatrix(const Matrix m) {
    // initialize matrix
    for (int i = 0; i < m.height * m.width; i++) {
        m.elements[i] = rand() % 100 + 1; // 1 - 100
        // m.elements[i] = 2;
    }

    // for (int i = 0; i < m.height; i++) {
    //     for (int j = 0; j < m.width; j++) {
    //         int index = i * m.width + j;
    //         cout << m.elements[index] << " ";
    //     }
    //     cout << endl;
    // }
}

int main() {
    // int a_height = 1024;
    // int a_width = 768;

    // int b_height = 768;
    // int b_width = 1024;

    int a_height = 20000;
    int a_width = 20000;

    int b_height = 20000;
    int b_width = 20000;

    int c_height = a_height;
    int c_width = b_width;

    size_t a_nBytes = a_height * a_width * sizeof(float);
    size_t b_nBytes = b_height * b_width * sizeof(float);
    size_t c_nBytes = c_height * c_width * sizeof(float);

    Matrix m_a = {a_height, a_width, (float *) malloc(a_nBytes)};
    Matrix m_b = {b_height, b_width, (float *) malloc(b_nBytes)};
    Matrix m_c = {c_height, c_width, (float *) malloc(c_nBytes)};

    // initialize matrix
    initializeMatrix(m_a);
    initializeMatrix(m_b);

    // initialize matrix on device
    Matrix d_a, d_b, d_c;
    d_a.height = m_a.height; d_a.width = m_a.width;
    d_b.height = m_b.height; d_b.width = m_b.width;
    d_c.height = m_c.height; d_c.width = m_c.width;

    CHECK(cudaMalloc((void **) &d_a.elements, a_nBytes));
    CHECK(cudaMalloc((void **) &d_b.elements, b_nBytes));
    CHECK(cudaMalloc((void **) &d_c.elements, c_nBytes));

    // copy matrix to device
    CHECK(cudaMemcpy(d_a.elements, m_a.elements, a_nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b.elements, m_b.elements, b_nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_c.elements, m_c.elements, c_nBytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((d_b.width + block.x - 1)/ block.x, (d_a.height + block.y - 1) / block.y);
    
    // start
    auto start = chrono::high_resolution_clock::now();
    // run kernel
    matrixMultiply<<<grid, block>>>(d_a, d_b, d_c);
    // synchronize device memory
    CHECK(cudaDeviceSynchronize());

    // end
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration = end - start;
    cout << "Global memory kernel execution time: " << duration.count() << " ms" << endl;

    // copy result from device to host
    CHECK(cudaMemcpy(m_c.elements, d_c.elements, c_nBytes, cudaMemcpyDeviceToHost));

    // free memory on cuda
    CHECK(cudaFree(d_a.elements));
    CHECK(cudaFree(d_b.elements));
    CHECK(cudaFree(d_c.elements));

    // for (int i = 0; i < m_c.height; i++) {
    //     for (int j = 0; j < m_c.width; j++) {
    //         int index_c = i * m_c.width + j;
    //         cout << m_c.elements[index_c] << " ";
    //     }
    //     cout << endl;
    // }

    // free memory on host
    free(m_a.elements);
    free(m_b.elements);
    free(m_c.elements);

    return 0;
}