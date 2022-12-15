#include <mpi.h>

int mpi_csr_matx_filling(int n, int m, double x_min, double x_max, double y_min, double y_max, 
        int *coords, int *dims_size, double *vals);


int mpi_right_filling(int n, int m, double x_min, double x_max, double y_min, double y_max,
        int *coords, int *dims_size, double *r_vec_rank);

int mat_vec_mul_sub(int n, int m, double *matx, double *w, double *r_vec, double *r_k);

int mpi_csr_mat_vec_mul(int n, int m, double *vals_rank, double *w_rank, 
        double *r_k, int *coords, int *dims_size, double *left_vector, 
        double *right_vector, double *bot_vector, double *top_vector);
int mat_vec_mul(int n, int m, double *matx, double *w, double *res);


double mpi_computing(int n, int m, double h1, double h2, double *vals_rank, double *w_rank, double *r_vec_rank,
        int *coords, int *dims_size, MPI_Comm *comm);

int vecsub(int n, int m, double *A, double *B, double *res);


double mpi_norm_e_matx_sqr(int n, int m, double h1, double h2, int *coords, int *dims_size, double *u, double *v);

double norm_e_sqr(int n, int m, double *vec);
double vec_scalar(int n, int m, double *A, double *B);
int rebalance(int n, int m, double *w, double tau_k, double *r_k);

int matx_print(int n, int m, double *matx);
int mpi_matx_print(int n, int m, double *matx, int *coords, int *dims_size);

double u(double x, double y);
double F(double x, double y);
double q(double x, double y);
double k(double x, double y);
double ksi(int i, int j, int n, int m, double x_min, double x_max, double y_min, double y_max);