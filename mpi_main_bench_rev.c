#include <mpi.h>
#include <stdio.h>
#include <math.h>

// reverse-mode AD entry point for 3 inputs
// must match your d_user_func signature exactly
extern void d_user_func(
    float x, float* dx,
    float y, float* dy,
    float z, float* dz,
    float dout
);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // each rank picks a distinct (x,y,z)
    float x = 1.0f + rank;
    float y = 2.0f + rank;
    float z = 3.0f + rank;

    // gradient accumulators
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    const int reps = 1000000;

    // synchronize and start timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // do the reverse-mode sweep reps times
    for (int i = 0; i < reps; ++i) {
        dx = dy = dz = 0.0f;
        d_user_func(x, &dx, y, &dy, z, &dz, 1.0f);
    }

    double t1 = MPI_Wtime();
    double time = t1 - t0;
    // total calls = reps * size
    double rate = (double)reps * size / time / 1e6;

    if (rank == 0) {
        printf("REV:  Cores=%d  Reps=%d  Time=%.6f s  Rate=%.3f M calls/s\n",
               size, reps, time, rate);
    }

    MPI_Finalize();
    return 0;
}
