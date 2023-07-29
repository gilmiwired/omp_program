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

void Compute_force(int part, vect_t forces[], struct particle_s all_curr, int n) {
   int k;
   double mg;
   vect_t f_part_k;
   double len, len_3, fact;

   forces[part][X] = forces[part][Y] = 0.0;
   for (k = 0; k < n; k++) {
      if (k != part) {
      /* Compute force on part due to k */
	       f_part_k[X] = all_curr.sx[part] - all_curr.sx[k];
	       f_part_k[Y] = all_curr.sy[part] - all_curr.sy[k];
         len = sqrt(f_part_k[X]*f_part_k[X] + f_part_k[Y]*f_part_k[Y]);
         len_3 = len*len*len;
         mg = -G*all_curr.m[part]*all_curr.m[k];
         fact = mg/len_3;
         f_part_k[X] *= fact;
         f_part_k[Y] *= fact;

         /* Add force in to total forces */
         forces[part][X] += f_part_k[X];
         forces[part][Y] += f_part_k[Y];
      }
   }
}  /* Compute_force */

void Update_part(int part, vect_t forces[], struct particle_s all_curr, int n, double delta_t) {
  double fact = delta_t/all_curr.m[part];

  all_curr.sx[part] += delta_t * all_curr.vx[part];
  all_curr.sy[part] += delta_t * all_curr.vy[part];
  all_curr.vx[part] += fact * forces[part][X];
  all_curr.vy[part] += fact * forces[part][Y];
}  /* Update_part */


void Output_state(double time, struct particle_s curr, int n, int my_rank, int local_n) {
   int part;
   if (my_rank == 0) { // only the process with rank 0 writes to the file
      FILE* fp;
      fp=fopen("log.csv","a");
      for (part = 0; part < n; part++) {
        fprintf(fp,"%3d, %10.3e, ", part, curr.sx[part]);
        fprintf(fp,"  %10.3e, ", curr.sy[part]);
        fprintf(fp,"  %10.3e, ", curr.vx[part]);
        fprintf(fp,"  %10.3e,", curr.vy[part]);
        fprintf(fp,"  %.2f\n", time);
      }
      fclose(fp);
   }
}  /* Output_state */


void Gen_init_cond(struct particle_s curr, int local_n, int local_start) {
  int part;
  double mass = 5.0e24;
  double gap = 1.0e5;
  double speed = 3.0e4;
  for (part = 0; part < local_n; part++) {
    curr.m[part] = mass;
    curr.sx[part] = (part + local_start) * gap;
    curr.sy[part] = 0.0;
    curr.vx[part] = 0.0;

    if ((part + local_start) % 2 == 0)
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
   struct particle_s curr, all_curr;
   vect_t* forces;
   vect_t* recv_forces;
   double start, finish;

   int my_rank,procs;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   MPI_Comm_size(MPI_COMM_WORLD,&procs);
   int local_n = n/procs;
   int local_start = my_rank * local_n;

   Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq);

   curr.m = malloc(local_n * sizeof(double));
   curr.sx = malloc(local_n * sizeof(double));
   curr.sy = malloc(local_n * sizeof(double));
   curr.vx = malloc(local_n * sizeof(double));
   curr.vy = malloc(local_n * sizeof(double));
   
   all_curr.m = malloc(n * sizeof(double));
   all_curr.sx = malloc(n * sizeof(double));
   all_curr.sy = malloc(n * sizeof(double));
   all_curr.vx = malloc(n * sizeof(double));
   all_curr.vy = malloc(n * sizeof(double));

   forces = malloc(local_n * sizeof(vect_t));
   recv_forces = malloc(n * sizeof(vect_t));

   Gen_init_cond(curr, local_n, local_start);

   memcpy(all_curr.m + local_start, curr.m, local_n * sizeof(double));
   memcpy(all_curr.sx + local_start, curr.sx, local_n * sizeof(double));
   memcpy(all_curr.sy + local_start, curr.sy, local_n * sizeof(double));
   memcpy(all_curr.vx + local_start, curr.vx, local_n * sizeof(double));
   memcpy(all_curr.vy + local_start, curr.vy, local_n * sizeof(double));

   Output_state(0, all_curr, n, my_rank, local_n);

   start=omp_get_wtime();

   for (step = 1; step <= n_steps; step++) {
      t = step*delta_t;

      #pragma omp parallel for
      for (part = 0; part < local_n; part++)
        Compute_force(part, forces, all_curr, n);

      MPI_Allgather(forces, local_n*DIM, MPI_DOUBLE, recv_forces, local_n*DIM, MPI_DOUBLE, MPI_COMM_WORLD);
      memcpy(forces, recv_forces, n*DIM*sizeof(double));

      #pragma omp parallel for
      for (part = 0; part < local_n; part++)
        Update_part(part, forces, all_curr, n, delta_t);

      memcpy(all_curr.m + local_start, curr.m, local_n * sizeof(double));
      memcpy(all_curr.sx + local_start, curr.sx, local_n * sizeof(double));
      memcpy(all_curr.sy + local_start, curr.sy, local_n * sizeof(double));
      memcpy(all_curr.vx + local_start, curr.vx, local_n * sizeof(double));
      memcpy(all_curr.vy + local_start, curr.vy, local_n * sizeof(double));

      if (step % output_freq == 0)
        Output_state(t, all_curr, n, my_rank, local_n);
   }

   finish=omp_get_wtime();
   printf("Elapsed time = %e seconds\n", finish-start);

   free(curr.m);
   free(curr.sx);
   free(curr.sy);
   free(curr.vx);
   free(curr.vy);
   
   free(all_curr.m);
   free(all_curr.sx);
   free(all_curr.sy);
   free(all_curr.vx);
   free(all_curr.vy);
   
   free(forces);
   free(recv_forces);

   MPI_Finalize();
   return 0;
}

