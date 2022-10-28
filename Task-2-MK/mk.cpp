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
    MPI_Status status;
    
    double precision = atof(argv[1]);
    
    int dots_each_iter = 1024;
    int block = dots_each_iter / comm_size;
    double xyz[3 * dots_each_iter];
    double xyz_local[3 * block];
    
    double sum_part = 0; 
    double sum = 0;
    double eps = precision + 1;
    int seed = 173;
    std::srand(seed);
    double true_sum = 3.14 / 6;
    double V = 2 * 2 * 1.0;
    true_sum = true_sum / V;
    int dot_count = 0;
    int criteria = 1;
    int time_count = 0;
    
    double starttime, endtime, resulttime;
    starttime = 0.0;
    endtime = 0.0;
    if (rank != 0) {
        for (int i = 0; i < 3 * block; i++) { 
             xyz_local[i] = 0.0;   
        }
    }
    while (criteria) {
        if (rank == 0) {
            for (int j = 1; j < comm_size; j++) {
                for (int k = 0; k < block; k++) {
                    xyz[3 * block * j + k] = -1.0 + 2.0 * std::rand()/RAND_MAX; // [-1.0, 1.0]
                    xyz[3 * block * j + block + k] = -1.0 + 2.0 * std::rand()/RAND_MAX; // [-1.0, 1.0]
                    xyz[3 * block * j + 2 * block + k] = 0.0 + 1.0 * std::rand()/RAND_MAX; // [0.0, 1.0]
                }
            }
        } 
        MPI_Scatter(xyz, 3 * block, MPI_DOUBLE, xyz_local, 3 * block, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            starttime = MPI_Wtime(); 
            for (int i = 0; i < block; i++) {
                if (xyz_local[i] * xyz_local[i] + xyz_local[block + i] * xyz_local[block + i] \
                        <= xyz_local[2 * block + i] * xyz_local[2 * block + i]) {
                    sum_part += sqrt(xyz_local[i] * xyz_local[i] + xyz_local[block + i] * xyz_local[block + i]);
                }
            }
            resulttime = MPI_Wtime() - starttime;
            endtime += resulttime;
        }
        MPI_Reduce(&sum_part, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            dot_count += (comm_size - 1) * block;
            eps = std::abs(true_sum - sum / dot_count);
                
            if (eps <= precision) {
                criteria = 0;
            }
        }
        MPI_Bcast(&criteria, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Reduce(&endtime, &resulttime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "value = " << V * sum / dot_count << "\n" << 
            "eps = " << eps << "\n" << 
            "dots = " << dot_count << "\n" << 
            "time = " << resulttime << "\n";
            //"seed = " << seed << "\n";
            //"true_value = " << true_sum << "\n" <<
            //"precision from console = " << precision << "\n" <<
    }
    MPI_Finalize();
    return 0;
}
