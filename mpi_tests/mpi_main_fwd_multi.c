#include <mpi.h>
#include <stdio.h>
#include <math.h>

// forward-mode AD entry point for f(x,y)
typedef struct { float val, dval; } _dfloat;
extern _dfloat d_user_func(_dfloat x, _dfloat y);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // each rank gets its own x,y
    _dfloat in_x = { .val = 1.0f + rank, .dval = 1.0f };  // seed ∂/∂x
    _dfloat in_y = { .val = 2.0f + rank, .dval = 0.0f };  // no seed for y

    // run AD
    _dfloat out = d_user_func(in_x, in_y);

    // out.val = f(x,y), out.dval = ∂f/∂x
    printf("Rank %d: x=%.2f, y=%.2f → f=%.6f, df/dx=%.6f\n",
           rank, in_x.val, in_y.val, out.val, out.dval);

    // If you also want ∂f/∂y, do a second call swapping the seeds:
    in_x.dval = 0.0f;
    in_y.dval = 1.0f;
    out = d_user_func(in_x, in_y);
    printf("Rank %d:         df/dy=%.6f\n", rank, out.dval);

    MPI_Finalize();
    return 0;
}
