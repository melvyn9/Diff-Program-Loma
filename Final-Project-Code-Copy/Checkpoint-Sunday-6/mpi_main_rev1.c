// --------------------------- Weak Scaling ------------------------------------------
// #include <mpi.h>
// #include <stdio.h>
// #include <math.h>
// #include <time.h>

// extern void d_user_func(float x, float* dx, float dout);

// int main(int argc, char** argv) {
//     MPI_Init(&argc, &argv);
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     const int reps = 1000000;
//     float x = 1.0f + rank;
//     float dx = 0.0f;
//     float dout = 1.0f;

//     double t0 = MPI_Wtime();
//     for (int i = 0; i < reps; ++i) {
//         dx = 0.0f;
//         d_user_func(x, &dx, dout);
//     }
//     double t1 = MPI_Wtime();

//     double time = t1 - t0;
//     double total_calls = (double)reps * size;         // total AD calls across all ranks
//     double rate       = total_calls / time / 1e6;     // in millions of calls/s

//     if (rank == 0)
//         printf("REV:  Cores=%d  Reps=%d  Time=%.6f s  Rate=%.3f M calls/s\n", size, reps, time, rate);

//     MPI_Finalize();
//     return 0;
// }

// --------------------------- Strong Scaling ------------------------------------------
#include <mpi.h>
#include <stdio.h>
#include <math.h>

// reverse-mode AD entry point
extern void d_user_func(float x, float* dx, float dout);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int TOTAL_REPS = 1000000;
    int reps_per_rank = TOTAL_REPS / size;

    float x = 1.0f + rank;
    float dx;
    float dout = 1.0f;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int i = 0; i < reps_per_rank; ++i) {
        dx = 0.0f;
        d_user_func(x, &dx, dout);
    }
    double t1 = MPI_Wtime();

    double local_time = t1 - t0, max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double rate = (double)TOTAL_REPS / max_time / 1e6;
        printf("REV-STRONG: Cores=%d  TotalReps=%d  Reps/Rank=%d\n"
               "             Time=%.6f s  Rate=%.3f M calls/s\n",
               size, TOTAL_REPS, reps_per_rank,
               max_time, rate);
    }

    MPI_Finalize();
    return 0;
}
