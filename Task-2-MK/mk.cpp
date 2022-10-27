#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <cmath>

int main(int argc, char *argv[]) {
    int comm_size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request request;
    
    double precision = atof(argv[1]);
    
    int dots_each_proc = 100;
    double *xyz = new double[3 * comm_size * dots_each_proc];
    double *xyz_local = new double[3 * dots_each_proc];
    
    double sum_part = 0; 
    double sum = 0;
    double eps = precision + 1;
    int seed = 758631;
    std::srand(seed);
    double true_sum = 3.14 / 6;
    double V = 2 * 2 * 1.0;
    true_sum = true_sum / V;
    int dot_count = 0;
    int criteria = 1;
    
    double starttime, endtime, resulttime;
    starttime = MPI_Wtime();
    
    if (rank != 0) {
        for (int i = 0; i < 3 * dots_each_proc; i++) { 
             xyz_local[i] = 0.0;   
        }
    }
    while (criteria) {
        if (rank == 0) {
            for (int j = 1; j < comm_size; j++) {
                for (int k = 0; k < dots_each_proc; k++) {
                    xyz[3 * dots_each_proc * j + k] = -1.0 + 2.0 * std::rand()/RAND_MAX; // [-1.0, 1.0]
                    xyz[3 * dots_each_proc * j + dots_each_proc + k] = -1.0 + 2.0 * std::rand()/RAND_MAX; // [-1.0, 1.0]
                    xyz[3 * dots_each_proc * j + 2 * dots_each_proc + k] = 0.0 + 1.0 * std::rand()/RAND_MAX; // [0.0, 1.0]
                }
            }
        } 
        MPI_Scatter(xyz, 3 * dots_each_proc, MPI_DOUBLE, xyz_local, 3 * dots_each_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            MPI_Status status;
            for (int i = 0; i < dots_each_proc; i++) {
                if (xyz_local[i] * xyz_local[i] + xyz_local[dots_each_proc + i] * xyz_local[dots_each_proc + i] \
                        <= xyz_local[2 * dots_each_proc + i] * xyz_local[2 * dots_each_proc + i]) {
                    sum_part += sqrt(xyz_local[i] * xyz_local[i] + xyz_local[dots_each_proc + i] * xyz_local[dots_each_proc + i]);
                }
            }
        }
        MPI_Reduce(&sum_part, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            dot_count += (comm_size - 1) * dots_each_proc;
            eps = std::abs(true_sum - sum / dot_count);
                
            if (eps <= precision) {
                criteria = 0;
            }
        }
        MPI_Bcast(&criteria, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    endtime = MPI_Wtime();
    endtime = endtime - starttime;
    MPI_Reduce(&endtime, &resulttime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {

        std::cout << "value = " << V * sum / dot_count << "\n" << 
            //"true_value = " << true_sum << "\n" <<
            //"precision from console = " << precision << "\n" <<
            "eps = " << eps << "\n" << 
            "dots = " << dot_count << "\n" << 
            "time = " << resulttime << "\n";
    }
    MPI_Finalize();
    return 0;
}
