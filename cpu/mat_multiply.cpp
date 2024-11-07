#include <iostream>
#include <chrono>

using namespace std;

struct Matrix{
    int height;
    int width;
    float *elements;
};

void matrixMultiply(const Matrix &a, const Matrix &b, Matrix &c) {
    for (int i = 0; i < a.height; i++) {
        for (int j = 0; j < b.width; j++) {
            float cvalue = 0;
            // c[i][j] = sum_k a[i][k] * b[k][j]
            for (int k = 0; k < a.width; k++) {
                int index_i = i * a.width + k;
                int index_j = k * b.width + j;
                cvalue += a.elements[index_i] * b.elements[index_j];
            }
            int index_c = i * c.width + j;
            c.elements[index_c] = cvalue;
        }
    }
}

void initializeMatrix(const Matrix m) {
    // initialize matrix
    for (int i = 0; i < m.height * m.width; i++) {
        // m.elements[i] = rand() % 100 + 1; // 1 - 100
        m.elements[i] = 2;
    }
}

int main() {
    int a_height = 1024;
    int a_width = 768;

    int b_height = 768;
    int b_width = 1024;

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

    // start
    auto start = chrono::high_resolution_clock::now();
    matrixMultiply(m_a, m_b, m_c);
    // end
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration = end - start;
    cout << "CPU execution time: " << duration.count() << " ms" << endl;

    // for (int i = 0; i < m_c.height; i++) {
    //     for (int j = 0; j < m_c.width; j++) {
    //         int index_c = i * m_c.width + j;
    //         cout << m_c.elements[index_c] << " ";
    //     }
    //     cout << endl;
    // }

    free(m_a.elements);
    free(m_b.elements);
    free(m_c.elements);

    return 0;
}