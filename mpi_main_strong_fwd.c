#include <mpi.h>
#include <stdio.h>
#include <math.h>

// forward-mode AD entry point for 3 inputs
typedef struct { float val, dval; } _dfloat;
extern _dfloat d_user_func(_dfloat x, _dfloat y, _dfloat z);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int total_reps = 1000000;
    const int reps = total_reps / size;

    // give each rank its own (x,y,z)
    _dfloat inx = { .val = 1.0f + rank, .dval = 1.0f };
    _dfloat iny = { .val = 2.0f + rank, .dval = 0.0f };
    _dfloat inz = { .val = 3.0f + rank, .dval = 0.0f };
    _dfloat out;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < reps; ++i) {
        // derivative w.r.t. x only; y,z seeds are zero
        out = d_user_func(inx, iny, inz);
    }
    double t1 = MPI_Wtime();

    double local_time = t1 - t0, max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double rate = (double)total_reps / max_time / 1e6;
        printf("FWD-STRONG-3D: Cores=%d  TotalReps=%d  Reps/Rank=%d\n",
               size, total_reps, reps);
        printf("               Time=%.6f s  Rate=%.3f M calls/s\n",
               max_time, rate);
    }

    MPI_Finalize();
    return 0;
}
