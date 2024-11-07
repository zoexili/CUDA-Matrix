#include <iostream>

using namespace std;

void vectorAdd(const float *a, const float *b, float *c, const int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    } 
}

int main() {

    int N = 3;
    size_t nBytes =  N * sizeof(float);

    // initialize 
    float *h_a = (float *) malloc(nBytes);
    float *h_b = (float *) malloc(nBytes);
    float *h_c = (float *) malloc(nBytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 3.0f;
    }

    vectorAdd(h_a, h_b, h_c, N);

    for (int i = 0; i < N; i++) {
        cout << h_c[i] << endl;
    }

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}