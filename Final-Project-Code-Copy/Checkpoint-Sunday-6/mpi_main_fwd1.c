// --------------------------- Weak Scaling ------------------------------------------
// // mpi_main_fwd.c
// #include <mpi.h>
// #include <stdio.h>
// #include <math.h>

// // forward-mode AD entry point
// // (d_user_func takes and returns an _dfloat struct)
// typedef struct { float val, dval; } _dfloat;
// extern _dfloat d_user_func(_dfloat x);

// int main(int argc, char** argv) {
//     MPI_Init(&argc, &argv);
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     const int REPS = 1000000;
//     _dfloat in  = { .val = 1.0f + rank, .dval = 1.0f };
//     _dfloat out;

//     // synchronize all ranks
//     MPI_Barrier(MPI_COMM_WORLD);
//     double t0 = MPI_Wtime();

//     // run REPS forward-mode AD calls
//     for (int i = 0; i < REPS; ++i) {
//         out = d_user_func(in);
//     }

//     double t1 = MPI_Wtime();
//     double local_time = t1 - t0, max_time;
//     MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         printf("FWD:  Cores=%d  Reps=%d  Time=%.6f s  Rate=%.3f M calls/s\n",
//                size, REPS, max_time, (REPS*1e-6*size)/max_time);
//     }

//     MPI_Finalize();
//     return 0;
// }

// --------------------------- Strong Scaling ------------------------------------------
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

    const int TOTAL_REPS = 1000000;
    int reps_per_rank = TOTAL_REPS / size;

    _dfloat in  = { .val = 1.0f + rank, .dval = 1.0f };
    _dfloat out;

    // synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // each rank does only reps_per_rank calls
    for (int i = 0; i < reps_per_rank; ++i) {
        out = d_user_func(in);
    }
    double t1 = MPI_Wtime();

    // gather the slowest rank’s time
    double local_time = t1 - t0, max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double rate = (double)TOTAL_REPS / max_time / 1e6;
        printf("FWD-STRONG: Cores=%d  TotalReps=%d  Reps/Rank=%d\n"
               "             Time=%.6f s  Rate=%.3f M calls/s\n",
               size, TOTAL_REPS, reps_per_rank,
               max_time, rate);
    }

    MPI_Finalize();
    return 0;
}
