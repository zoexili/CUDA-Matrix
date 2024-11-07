#include<iostream>

using namespace std;

void matrixAdd(const float *a, const float *b, float *c, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int index = i * cols + j;
            c[index] = a[index] + b[index];
        }
    }
}

int main() {
    int n_rows = 5;
    int n_cols = 3;
    size_t nBytes =  n_rows * n_cols * sizeof(float);

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

    matrixAdd(h_a, h_b, h_c, n_rows, n_cols);

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            int index = i * n_cols + j;
            cout << h_c[index] << " ";
        }
        cout << endl;
    }
}