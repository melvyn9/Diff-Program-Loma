
#include <mpi.h>
#include <stdio.h>

// reverse-mode AD entry point
extern void d_user_func(float x, float* dx, float dout);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float x = 1.0f + rank;
    float grad = 0.0f;
    printf("Rank %d: x=%.2f seed=1.00\n", rank, x);
    d_user_func(x, &grad, 1.0f);
    printf("Rank %d: dy=%.6f\n", rank, grad);

    float total = 0.0f;
    MPI_Allreduce(&grad, &total, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) printf("Global gradient: %.6f\n", total);

    MPI_Finalize();
    return 0;
}
