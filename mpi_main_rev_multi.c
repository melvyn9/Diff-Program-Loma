#include <mpi.h>
#include <stdio.h>
#include <math.h>

// reverse-mode AD entry point for f(x,y)
extern void d_user_func(float x, float y,
                       float* dx, float* dy,
                       float dout);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    float x = 1.0f + rank;
    float y = 2.0f + rank;
    float dx = 0.0f, dy = 0.0f;

    // run AD
    d_user_func(x, y, &dx, &dy, 1.0f);

    printf("Rank %d: x=%.2f, y=%.2f → df/dx=%.6f, df/dy=%.6f\n",
           rank, x, y, dx, dy);

    // Optionally reduce across ranks if you want global sums:
    float gdx=0.0f, gdy=0.0f;
    MPI_Allreduce(&dx, &gdx, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&dy, &gdy, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (rank==0) {
      printf("Global ∂f/∂x = %.6f, Global ∂f/∂y = %.6f\n", gdx, gdy);
    }

    MPI_Finalize();
    return 0;
}
