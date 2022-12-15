#include <iostream>
#include <computing.hpp>
#include <mpi.h>
#include <omp.h>

int main(int argc, char **argv) {
    int comm_size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dims_size[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2] = {0, 0};
    MPI_Comm comm;
    MPI_Dims_create(comm_size, 2, dims_size);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims_size, periods, 1, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coords);

    int n = atoi(argv[1]); // all dots
    int m = atoi(argv[2]); // all dots
    double x_min = -2;
    double x_max = 3;
    double y_min = -1;
    double y_max = 4;
    double h1 = (x_max - x_min) / (n - 1);
    double h2 = (y_max - y_min) / (m - 1);
    double /**r_vec, *w, *vals,*/ *r_vec_rank, *w_rank, *vals_rank;
    // double *matx = new double[n*m * n*m];

    r_vec_rank = new double[n/dims_size[0] * m/dims_size[1]];
    w_rank = new double[n/dims_size[0] * m/dims_size[1]];
    vals_rank = new double[5 * n/dims_size[0] * m/dims_size[1]];

    // r_vec = new double[n*m];
    // w = new double[n*m];
    // vals = new double[5*n*m];
    // for (int j = 0; j < m; j++) {
    //     for (int i = 0; i < n; i++) {
    //         w[j*n + i] = u(x_min + i*h1, y_min + j*h2);
    //     }
    // }
        
    // for (int i = 0; i < n*m; i++) {
    //     w[i] = 0.0;
    // }
    // for (int i = 0; i < n; i++) {
    //     w[i] = 0.0;//u(x_min + i*h1, 0); 
    // }
    int curr_i, curr_j;
    for (int j = 0; j < m / dims_size[1]; j++) {
        for (int i = 0; i < n / dims_size[0]; i++) {
            curr_i = coords[0] * n / dims_size[0] + i;
            curr_j = coords[1] * m / dims_size[1] + j;
            // if (curr_j == 0) {
            //     w_rank[i] = 0.0;
            // } else {
                w_rank[j*n/dims_size[0] + i] = 0;//u(x_min + curr_i*h1, y_min + curr_j*h2);
            // }
        }
    }

    double start_time = MPI_Wtime();
    mpi_csr_matx_filling(n, m, x_min, x_max, y_min, y_max, coords, dims_size, vals_rank);
    // if (rank == 1) {
    // for (int j = 0; j < m / dims_size[1]; j++) {
    //     for (int i = 0; i < n / dims_size[0]; i++) {
    //         std::cout << rank << " " << 
    //         vals_rank[5*(j * n / dims_size[0] + i)] << " " <<
    //         vals_rank[5*(j * n / dims_size[0] + i) + 1] << " " <<
    //         vals_rank[5*(j * n / dims_size[0] + i) + 2] << " " <<
    //         vals_rank[5*(j * n / dims_size[0] + i) + 3] << " " <<
    //         vals_rank[5*(j * n / dims_size[0] + i) + 4] << " " <<    
    //         coords[0] * n / dims_size[0] + i << " " << coords[1] * m / dims_size[1] + j << std::endl;
    //     }
    // }
    // }

    mpi_right_filling(n, m, x_min, x_max, y_min, y_max, coords, dims_size, r_vec_rank);
    // mpi_matx_print(n/dims_size[0], m/dims_size[1], r_vec_rank, coords, dims_size);
    // if (rank == 0) {
    // for (int j = 0; j < m / dims_size[1]; j++) {
    //     for (int i = 0; i < n / dims_size[0]; i++) {
    //         std::cout << rank << " " << 
    //         r_vec_rank[j * n / dims_size[0] + i] << " " <<   
    //         coords[0] * n / dims_size[0] + i << " " << coords[1] * m / dims_size[1] + j << std::endl;
    //     }
    // }
    // }
    
    double error;
    error = mpi_computing(n, m, h1, h2, vals_rank, w_rank, r_vec_rank, coords, dims_size, &comm);
    double end_time = MPI_Wtime();
    double result_time, time = end_time - start_time;
    MPI_Reduce(&time, &result_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    // for (int j = 0; j < m / dims_size[1]; j++) {
    //     for (int i = 0; i < n / dims_size[0]; i++) {
    //         curr_i = coords[0] * n / dims_size[0] + i;
    //         curr_j = coords[1] * m / dims_size[1] + j;
    //         // if (curr_j == 0) {
    //         //     w_rank[i] = 0.0;
    //         // } else {
    //             std::cout << w_rank[j*n/dims_size[0] + i] - u(x_min + curr_i*h1, y_min + curr_j*h2) \
    //             << " " << curr_i << " " << curr_j << std::endl;
    //         // }
    //     }
    // }
    if (rank == 0) {
        std::cout << "TIME = " << result_time << " " << error << std::endl;
    }

    MPI_Barrier(comm);
    for (int j = 0; j < m / dims_size[1]; j++) {
        for (int i = 0; i < n / dims_size[0]; i++) {
            curr_i = coords[0] * n / dims_size[0] + i;
            curr_j = coords[1] * m / dims_size[1] + j;
            if (curr_j == 0) {
                w_rank[j*n/dims_size[0] + i] = u(x_min + curr_i*h1, y_min + curr_j*h2);
            }
        }
    }

    // MPI_Barrier(comm);

    // for (int j = 0; j < m / dims_size[1]; j++) {
    //     for (int i = 0; i < n / dims_size[0]; i++) {
    //         curr_i = coords[0] * n / dims_size[0] + i;
    //         curr_j = coords[1] * m / dims_size[1] + j;
    //         std::cout << x_min + h1 * curr_i << " " \
    //         << y_min + h2 * curr_j << " " \
    //         << w_rank[j*n/dims_size[0] + i] << " " \
    //         << curr_j * n + curr_i << std::endl;
    //     }
    // }

    delete[] vals_rank;
    delete[] r_vec_rank;
    delete[] w_rank;
    
    MPI_Finalize();
}