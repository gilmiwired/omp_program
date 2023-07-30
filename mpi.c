#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#define DIM 2  /* Two-dimensional system */
#define X 0    /* x-coordinate subscript */
#define Y 1    /* y-coordinate subscript */

const double G = 6.673e-11;  /* Gravitational constant. */
typedef double vect_t[DIM];  /* Vector type for position, etc. */

// added type..
struct particle_s {
  double *m;        /* Mass     */
  double *sx,*sy;  /* Position */
  double *vx,*vy;  /* Velocity */
};


void Get_args(int argc, char* argv[], int* n_p, int* n_steps_p,
          double* delta_t_p, int* output_freq_p) {
  *n_p       = strtol(argv[1], NULL, 10);
  *n_steps_p = strtol(argv[2], NULL, 10);
  *delta_t_p = strtod(argv[3], NULL);
  *output_freq_p = strtol(argv[4], NULL, 10);
}  /* Get_args */

void Compute_force(int part, vect_t forces[], struct particle_s curr,
      int n) {
   int k;
   double mg;
   vect_t f_part_k;
   double len, len_3, fact;

   forces[part][X] = forces[part][Y] = 0.0;
   for (k = 0; k < n; k++) {
      if (k != part) {
      /* Compute force on part due to k */
           f_part_k[X] = curr.sx[part] - curr.sx[k];
           f_part_k[Y] = curr.sy[part] - curr.sy[k];
         len = sqrt(f_part_k[X]*f_part_k[X] + f_part_k[Y]*f_part_k[Y]);
         len_3 = len*len*len;
         mg = -G*curr.m[part]*curr.m[k];
         fact = mg/len_3;
         f_part_k[X] *= fact;
         f_part_k[Y] *= fact;
   
         /* Add force in to total forces */
         forces[part][X] += f_part_k[X];
         forces[part][Y] += f_part_k[Y];
      }
   }
}  /* Compute_force */

void Update_part(int part, vect_t forces[], struct particle_s curr,
      int n, double delta_t) {
  double fact = delta_t/curr.m[part];

  curr.sx[part] += delta_t * curr.vx[part];
  curr.sy[part] += delta_t * curr.vy[part];
  curr.vx[part] += fact * forces[part][X];
  curr.vy[part] += fact * forces[part][Y];
}  /* Update_part */

void Output_state(double time, struct particle_s curr, int n) {
   int part;
   FILE* fp;
   fp=fopen("log.csv","a");
   for (part = 0; part < n; part++) {
     //     printf("%.3f ", curr[part].m);
     fprintf(fp,"%3d, %10.3e, ", part, curr.sx[part]);
     fprintf(fp,"  %10.3e, ", curr.sy[part]);
     fprintf(fp,"  %10.3e, ", curr.vx[part]);
     fprintf(fp,"  %10.3e,", curr.vy[part]);
     fprintf(fp,"  %.2f\n", time);
   }
   //   printf("\n");
   fclose(fp);
}  /* Output_state */

void Gen_init_cond(struct particle_s curr, int n) {
  int part;
  double mass = 5.0e24;
  double gap = 1.0e5;
  double speed = 3.0e4;
  for (part = 0; part < n; part++) {
    curr.m[part] = mass;
    curr.sx[part] = part*gap;
    curr.sy[part] = 0.0;
    curr.vx[part] = 0.0;

    if (part % 2 == 0)
      curr.vy[part] = speed;
    else
      curr.vy[part] = -speed;
  }
}  /* Gen_init_cond */


int main(int argc, char* argv[]) {
    int n;
    int n_steps;
    int step;
    int part;
    int output_freq;
    double delta_t;
    double t;
    vect_t* forces;
    double start, finish;

    int my_rank, procs;
    struct particle_s curr;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq);
    curr.m = malloc(n * sizeof(double));
    curr.sx = malloc(n * sizeof(double));
    curr.sy = malloc(n * sizeof(double));
    curr.vx = malloc(n * sizeof(double));
    curr.vy = malloc(n * sizeof(double));

    // Allocate separate buffers for gathering
    double* tmp_sx = malloc(n * sizeof(double));
    double* tmp_sy = malloc(n * sizeof(double));

    forces = malloc(n * sizeof(vect_t));
    Gen_init_cond(curr, n);

    int local_n = n / procs;
    int local_start = my_rank * local_n;
    int local_end = (my_rank != procs - 1) ? local_start + local_n : n;

    MPI_Barrier(MPI_COMM_WORLD);
    start = omp_get_wtime();
    if (my_rank == 0) {
        Output_state(0, curr, n);
    }

    for (step = 1; step <= n_steps; step++) {
        t = step * delta_t;
        for (part = local_start; part < local_end; part++)
            Compute_force(part, forces, curr, n);
        for (part = local_start; part < local_end; part++)
            Update_part(part, forces, curr, n, delta_t);
        if (step % output_freq == 0 && my_rank == 0)
            Output_state(t, curr, n);

        MPI_Allgather(curr.sx + local_start, local_n, MPI_DOUBLE, tmp_sx, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(curr.sy + local_start, local_n, MPI_DOUBLE, tmp_sy, local_n, MPI_DOUBLE, MPI_COMM_WORLD);

        // Copy gathered data back to original buffers
        memcpy(curr.sx, tmp_sx, n * sizeof(double));
        memcpy(curr.sy, tmp_sy, n * sizeof(double));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    finish = omp_get_wtime();
    if (my_rank == 0) {
        printf("Elapsed time = %e seconds\n", finish - start);
    }

    free(curr.m);
    free(curr.sx);
    free(curr.sy);
    free(curr.vx);
    free(curr.vy);
    free(forces);

    free(tmp_sx);
    free(tmp_sy);

    MPI_Finalize();
    return 0;
}

