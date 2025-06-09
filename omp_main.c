// omp_main.c
#include <stdio.h>
#include <omp.h>

// The reverse‐mode AD result: value + gradient
typedef struct {
    float val;
    float dval;
} _dfloat;

// This comes from your shared library d_mysin.so
extern void d_user_func(float x, float* dx, float dout);

int main() {
    const int N = 8;
    float inputs[N];
    float grads[N];

    // Prepare a few test inputs: 1.0, 2.0, …, 8.0
    for (int i = 0; i < N; ++i) {
        inputs[i] = 1.0f + i;
        grads[i]  = 0.0f;
    }

    // Parallel reverse‐mode AD over the batch
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        // seed ∂output/∂output = 1.0f
        d_user_func(inputs[i], &grads[i], 1.0f);
        printf("Thread %d: x=%.2f → dy=%.6f\n",
               omp_get_thread_num(), inputs[i], grads[i]);
    }

    return 0;
}
