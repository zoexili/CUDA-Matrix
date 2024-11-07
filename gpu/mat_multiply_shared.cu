#include<iostream>
#include<cuda_runtime.h>
#include<chrono>

using namespace std;

// tiling and shared memory
#define TILE_SIZE 100
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
    // shared memory for tile A and tile B
    __shared__ float as[TILE_SIZE][TILE_SIZE];
    __shared__ float bs[TILE_SIZE][TILE_SIZE];

    // calculate the global row and column index of the element
    int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    int global_j = blockIdx.y * blockDim.y + threadIdx.y;
    float cvalue = 0.0f;

    // row and column index in csub
    int i = threadIdx.x;
    int j = threadIdx.y;

    // loop over the tiles of the input matrices
    // c[i][j] = sum_k a[i][k] * b[k][j]
    // a.width / tile size and b.height / tile size
    for (int m = 0; m < (a. width + TILE_SIZE - 1)/ TILE_SIZE; m++) {
        // load a into shared memory as
        // as[i * TILE_SIZE + j]
        if (i < a.height && (m * TILE_SIZE + j) < a.width) {
            as[i][j] = a.elements[global_i * a.width + m * TILE_SIZE + j];
        }
        // when matrix dimensions are not multiples of TILE_SIZE, some threads will access elements 
        // out of matrix boundary. Set out-of-bound element to 0. 
        else {
            as[i][j] = 0.0f;
        }
        // load b into shared memory bs
        if (j < b.width && (m * TILE_SIZE + i) < b.height) {
            bs[i][j] = b.elements[(m * TILE_SIZE + i) * b.width + global_j]; 
        }
        else {
            bs[i][j] = 0.0f;
        }
        __syncthreads(); // synchornize - all threads load elements

        // compute the partial result
        for (int k = 0; k < TILE_SIZE; k++) {
            cvalue += as[i][k] * bs[k][j];
        }
        __syncthreads(); // synchronize - all threads complete computation
    }

    // write result back to global memory
    if (global_i < c.height && global_j < c.width) {
        c.elements[global_i * c.width + global_j] = cvalue;
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

    int a_height = 1000;
    int a_width = 1000;

    int b_height = 1000;
    int b_width = 1000;

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

    dim3 block(TILE_SIZE, TILE_SIZE);
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
    cout << "Shared memory kernel execution time: " << duration.count() << " ms" << endl;

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