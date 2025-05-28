#include <stdio.h>
#include <mpi.h>

// Declare the differentiated function and type
typedef struct {
    float val;
    float dval;
} _dfloat;

_dfloat d_square(_dfloat x);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float x = (float)(5);
    _dfloat input;
    input.val = x;
    input.dval = 1.0f;

    printf("Rank %d: Calling d_square(x=%.2f, dx=1.00)\n", rank, input.val);
    _dfloat result = d_square(input);
    printf("Rank %d: Got dy = %f\n", rank, result.dval);

    float global_grad;
    MPI_Reduce(&result.dval, &global_grad, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global gradient: %f\n", global_grad);
    }

    MPI_Finalize();
    return 0;
}
