// mpi_main_bench_fwd.c
#include <mpi.h>
#include <stdio.h>
#include <math.h>

// forward-mode AD entry point
typedef struct { float val, dval; } _dfloat;
extern _dfloat d_user_func(_dfloat x);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int reps = 1000000;
    // give each rank a different x
    _dfloat in  = { .val = 1.0f + rank, .dval = 1.0f };
    _dfloat out;

    // synchronize all ranks, then time the loop
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < reps; ++i) {
        out = d_user_func(in);
    }
    double t1 = MPI_Wtime();

    // find the slowest rank
    double local_time = t1 - t0, max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double rate = (double)reps * size / max_time / 1e6;
        printf("FWD:  Cores=%d  Reps=%d  Time=%.6f s  Rate=%.3f M calls/s\n",
               size, reps, max_time, rate);
    }

    MPI_Finalize();
    return 0;
}
