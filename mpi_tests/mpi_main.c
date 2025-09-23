#include <mpi.h>
#include <stdio.h>
#include <math.h>

// auto-generated for 3 inputs
static const char* arg_names[] = { "x", "y", "z" };

// Reverse‐mode AD entry point
extern void d_user_func(float x, float y, float z, float* dx, float* dy, float* dz, float dout);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float x = 1.0f + rank + 0;
    float dx = 0.0f;
    float y = 1.0f + rank + 1;
    float dy = 0.0f;
    float z = 1.0f + rank + 2;
    float dz = 0.0f;
    float dout = 1.0f;

    // single reverse sweep
    MPI_Barrier(MPI_COMM_WORLD);
    d_user_func(x, y, z, &dx, &dy, &dz, dout);

    printf("Rank %d: df/dx=%.6f df/dy=%.6f df/dz=%.6f\n", rank, dx, dy, dz);

    MPI_Finalize();
    return 0;
}