#include <mpi.h>
#include <stdio.h>
#include <math.h>

// reverse-mode AD entry point for 3 inputs
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

    const int total_reps = 1000000;
    const int reps = total_reps / size;

    float x  = 1.0f + rank;
    float y  = 2.0f + rank;
    float z  = 3.0f + rank;
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < reps; ++i) {
        dx = dy = dz = 0.0f;
        d_user_func(x, &dx, y, &dy, z, &dz, 1.0f);
    }
    double t1 = MPI_Wtime();

    double local_time = t1 - t0, max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double rate = (double)total_reps / max_time / 1e6;
        printf("REV-STRONG-3D: Cores=%d  TotalReps=%d  Reps/Rank=%d\n",
               size, total_reps, reps);
        printf("               Time=%.6f s  Rate=%.3f M calls/s\n",
               max_time, rate);
    }

    MPI_Finalize();
    return 0;
}
