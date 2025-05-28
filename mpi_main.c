#include <mpi.h>
#include <stdio.h>

typedef struct {
    float val;
    float dval;
} _dfloat;

extern _dfloat d_cube(_dfloat x);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float x_val = 1.0f + rank;
    _dfloat x;
    x.val = x_val;
    x.dval = 1.0f;  // seed derivative

    _dfloat result = d_cube(x);

    float dy = result.dval;

    float total_dy = 0.0f;
    MPI_Allreduce(&dy, &total_dy, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    printf("Rank %d: d_cube(x=%.2f, dx=1.0) => dy=%.6f\n", rank, x_val, dy);

    if (rank == 0) {
        printf("Global gradient: %f\n", total_dy);
    }

    MPI_Finalize();
    return 0;
}
