#include <iostream>
#include <cmath>
#include "computing.hpp"
#include <mpi.h>
#include <omp.h>

double u(double x, double y) {
    return 2.0 / (1 + x*x + y*y);
}

double k(double x, double y) {
    return 4 + x;
}

double q(double x, double y) {
    return 1.0;
}

double F(double x, double y) {
    return (-4*x - 32) / (1 + x*x + y*y) / (1 + x*x + y*y) + \
            16 * (x + 4) / (1 + x*x + y*y) / (1 + x*x + y*y) / (1 + x*x + y*y) + \
            2 / (1 + x*x + y*y);
}

int matx_print(int n, int m, double *matx) {
    // print by vertical cols 
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            // std::cout << "i=" << i << " j=" << j << " " << matx[i*cols + j] << " ";
            std::cout << matx[j*n + i] << std::endl;
        }
        std::cout << std::endl;
    }
    return 0;
}

int mpi_matx_print(int n, int m, double *matx, int *coords, int *dims_size)
{
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            // std::cout << "i=" << i << " j=" << j << " " << matx[i*cols + j] << " ";
            std::cout << matx[j*n + i] << " " << coords[0] * n + i << " " << coords[1] * m + j << std::endl; //<<  " " << coords[0] << " " << coords[1] << std::endl;
        }
    }
    return 0;
}

double ksi(int i, int j, int n, int m, double x_min, double x_max, double y_min, double y_max) {
    double h1 = (x_max - x_min) / (n - 1);
    double h2 = (y_max - y_min) / (m - 1);
    double x = x_min + i * h1;
    double y = y_min + j * h2;
    if ((i == 0) && (j == 0)) {
        return 2 * (h2 * (3*x*x + y*y + 8*x + 1) + h1 * (8*y + 2*x*y + x*x + y*y + 1)) / (h1 + h2) / (1 + x*x + y*y) / (1 + x*x + y*y);
        // A1, B1
        // return (4*x*x + 2*y*y + 8*x + 2 + 8*y + 2*x*y) / (1 + x*x + y*y) / (1 + x*x + y*y); 
    } else if ((i == 0) && (j == m - 1)) {
        return 2 * (h2 * (3*x*x + y*y + 8*x + 1) + h1 * (-8*y - 2*x*y + x*x + y*y + 1)) / (h1 + h2) / (1 + x*x + y*y) / (1 + x*x + y*y);
        // A1, B2
        // return (4*x*x + 2*y*y + 8*x + 2 - 8*y - 2*x*y) / (1 + x*x + y*y) / (1 + x*x + y*y);
    } else if ((i == n - 1) && (j == 0)) {
        return 2 * (h1 * (8*y + 2*x*y + x*x + y*y + 1) + h2 * (-x*x +y*y - 8*x + 1)) / (h1 + h2) / (1 + x*x + y*y) / (1 + x*x + y*y);
        // A2, B1
        // return (2*y*y - 8*x + 2 + 8*y + 2*x*y) / (1 + x*x + y*y) / (1 + x*x + y*y);
    } else if ((i == n - 1) && (j == m - 1)) {
        return 2 * (h1 * (-8*y - 2*x*y + x*x + y*y + 1) + h2 * (-x*x +y*y - 8*x + 1)) / (h1 + h2) / (1 + x*x + y*y) / (1 + x*x + y*y);
        // A2, B2
        // return (2*y*y - 8*x + 2 - 8*y - 2*x*y) / (1 + x*x + y*y) / (1 + x*x + y*y);
    } else if (i == 0) {
        // left
        return 2 * (3*x*x + y*y + 8*x + 1) / (1 + x*x + y*y) / (1 + x*x + y*y);
    } else if (i == n - 1) {
        // right
        return 2 * (-x*x + y*y - 8*x + 1) / (1 + x*x + y*y) / (1 + x*x + y*y);
    } else if (j == 0) {
        // bot
        return 2 * (8*y + 2*x*y + x*x + y*y + 1) / (1 + x*x + y*y) / (1 + x*x + y*y);
    } else if (j == m - 1) {
        // top
        return 2 * (-8*y - 2*x*y + x*x + y*y + 1) / (1 + x*x + y*y) / (1 + x*x + y*y);
    }
    return 0.0;
}

double phi(double x, double y) {
    return u(x, y);
}



int mpi_csr_matx_filling(int n, int m, double x_min, double x_max, double y_min, double y_max, int *coords, int *dims_size, double *vals)
{
    double h1 = (x_max - x_min) / (n - 1);
    double h2 = (y_max - y_min) / (m - 1);
    double alpha_l = 1.0;
    double alpha_r = 1.0;
    double alpha_t = 1.0;
    double alpha_b = 1.0;
    double a_left, a_right, b_down, b_up;
    int curr_i, curr_j;
    int n_curr = n / dims_size[0];
    int m_curr = m / dims_size[1];
    #pragma omp parallel for collapse(2) num_threads(4)
    for (int j = 0; j < m_curr; j++) {
        for (int i = 0; i < n_curr; i++) {
            curr_i = coords[0] * n_curr + i;
            curr_j = coords[1] * m_curr + j;
            a_left = 1 / h1 / h1 * k(x_min + (curr_i - 0.5) * h1, y_min + curr_j * h2);
            a_right = 1 / h1 / h1 * k(x_min + (curr_i + 0.5) * h1, y_min + curr_j * h2);
            b_down = 1 / h2 / h2 * k(x_min + curr_i * h1, y_min + (curr_j - 0.5) * h2);
            b_up = 1 / h2 / h2 * k(x_min + curr_i * h1, y_min + (curr_j + 0.5) * h2);
            if ((curr_i == 0) && (curr_j != m - 1) && (curr_j > 1)) {
                // left
                vals[5*(j*n_curr + i)] = -1 * b_down;
                vals[5*(j*n_curr + i) + 1] = 0.0;
                vals[5*(j*n_curr + i) + 2] = \
                    2 * a_right + \
                    (q(x_min + curr_i * h1, y_min + curr_j * h2) + 2 * alpha_l / h1) + \
                    b_up + b_down;
                vals[5*(j*n_curr + i) + 3] = -2 * a_right;
                vals[5*(j*n_curr + i) + 4] = -1 * b_up;
            } else if ((curr_i == n - 1) && (curr_j != m - 1) && (curr_j > 1)) {
                // right
                vals[5*(j*n_curr + i)] = -1 * b_down;
                vals[5*(j*n_curr + i) + 4] = -1 * b_up;
                vals[5*(j*n_curr + i) + 2] = \
                    2 * a_left + \
                    (q(x_min + curr_i * h1, y_min + curr_j * h2) + 2 * alpha_r / h1) + \
                    b_down + b_up;
                vals[5*(j*n_curr + i) + 1] = -2 * a_left;
                vals[5*(j*n_curr + i) + 3] = 0.0;
            } else if ((curr_j == 1) && (curr_i != n - 1) && (curr_i != 0)) {
                // bottom
                vals[5*(j*n_curr + i)] = 0.0;
                vals[5*(j*n_curr + i) + 1] = -1 * a_left;
                vals[5*(j*n_curr + i) + 3] = -1 * a_right;
                vals[5*(j*n_curr + i) + 2] = \
                    a_left + a_right + \
                    b_up + b_down + \
                    q(x_min + curr_i * h1, y_min + curr_j * h2);
                vals[5*(j*n_curr + i) + 4] = -1 * b_up;
            } else if ((curr_j == m - 1) && (curr_i != n - 1) && (curr_i != 0)) {
                // top 
                vals[5*(j*n_curr + i) + 4] = 0.0;
                vals[5*(j*n_curr + i) + 1] = -1 * a_left;
                vals[5*(j*n_curr + i) + 3] = -1 * a_right;
                vals[5*(j*n_curr + i) + 2] = \
                    2 * b_down + \
                    (q(x_min + curr_i * h1, y_min + curr_j * h2) + 2 * alpha_t / h2) + \
                    a_left + a_right;
                vals[5*(j*n_curr + i)] = -2 * b_down;
            } else if ((curr_i == n - 1) && (curr_j == m - 1)) {
                // A2, B2
                vals[5*(j*n_curr + i) + 3] = 0.0;
                vals[5*(j*n_curr + i) + 4] = 0.0;
                vals[5*(j*n_curr + i) + 1] = -2 * a_left;
                vals[5*(j*n_curr + i)] = -2 * b_down;
                vals[5*(j*n_curr + i) + 2] = \
                    2 * a_left + \
                    2 * b_down + \
                    (q(x_min + curr_i * h1, y_min + curr_j * h2) + 2 * alpha_r / h1 + 2 * alpha_t / h2);
            } else if ((curr_i == n - 1) && (curr_j == 1)) {
                // A2, B1 + 1
                vals[5*(j*n_curr + i)] = 0.0;
                vals[5*(j*n_curr + i) + 3] = 0.0;
                vals[5*(j*n_curr + i) + 1] = -2 * a_left;
                vals[5*(j*n_curr + i) + 2] = \
                    2 * a_left + \
                    (q(x_min + curr_i * h1, y_min + curr_j * h2) + 2 * alpha_r / h1) + \
                    b_down + b_up;
                vals[5*(j*n_curr + i) + 4] = -1 * b_up;
            } else if ((curr_i == 0) && (curr_j == m - 1)) {
                // A1, B2
                vals[5*(j*n_curr + i) + 1] = 0.0;
                vals[5*(j*n_curr + i) + 4] = 0.0;
                vals[5*(j*n_curr + i) + 3] = -2 * a_right;
                vals[5*(j*n_curr + i)] = -2 * b_down;
                vals[5*(j*n_curr + i) + 2] = \
                    2 * a_right + \
                    2 * b_down + \
                    (q(x_min + curr_i * h1, y_min + curr_j * h2) + 2 * alpha_l / h1 + 2 * alpha_t / h2);
            } else if ((curr_i == 0) && (curr_j == 1)) { 
                // A1, B1 + 1
                vals[5*(j*n_curr + i)] = 0.0;
                vals[5*(j*n_curr + i) + 1] = 0.0;
                vals[5*(j*n_curr + i) + 2] = \
                    2 * a_right + \
                    (q(x_min + curr_i * h1, y_min + curr_j * h2) + 2 * alpha_l / h1) + \
                    b_down + b_up;
                vals[5*(j*n_curr + i) + 3] = -2 * a_right;
                vals[5*(j*n_curr + i) + 4] = -1 * b_up;
            } else if (curr_j == 0) {
                // bot bottom
                vals[5*(j*n_curr + i)] = 0.0;
                vals[5*(j*n_curr + i) + 1] = 0.0;
                vals[5*(j*n_curr + i) + 2] = 0.0;
                vals[5*(j*n_curr + i) + 3] = 0.0;
                vals[5*(j*n_curr + i) + 4] = 0.0;
                // matx[(j*n + i) * n*m + j*n + i] = 0; //u(x_min + i * h1, y_min + j * h2);
            } else {
                // inner
                vals[5*(j*n_curr + i) + 1] = -1 * a_left;
                vals[5*(j*n_curr + i) + 3] = -1 * a_right;
                vals[5*(j*n_curr + i) + 2] = \
                    a_left + a_right + \
                    b_down + b_up + \
                    q(x_min + curr_i * h1, y_min + curr_j * h2);
                vals[5*(j*n_curr + i)] = -1 * b_down;
                vals[5*(j*n_curr + i) + 4] = -1 * b_up;
            }
        }
    }
    return 0;
}


int mpi_right_filling(int n, int m, double x_min, double x_max, double y_min, double y_max, int *coords, int *dims_size, double *r_vec_rank) {
    double h1 = (x_max - x_min) / (n - 1);
    double h2 = (y_max - y_min) / (m - 1);
    int curr_i, curr_j;
    int n_curr = n / dims_size[0];
    int m_curr = m / dims_size[1];
    double F_i_j, ksi_i_j;
    #pragma omp parallel for collapse(2) num_threads(4)
    for (int j = 0; j < m_curr; j++) {
        for (int i = 0; i < n_curr; i++) {
            curr_i = coords[0] * n_curr + i;
            curr_j = coords[1] * m_curr + j;
            F_i_j = F(x_min + curr_i * h1, y_min + curr_j * h2);
            ksi_i_j = ksi(curr_i, curr_j, n, m, x_min, x_max, y_min, y_max);
            if ((curr_i == 0) && (curr_j < m - 1) && (curr_j > 1)) {
                // left
                r_vec_rank[j*n_curr + i] = \
                    F_i_j + \
                    2 / h1 * ksi_i_j;
            } else if ((curr_i == n - 1) && (curr_j < m - 1) && (curr_j > 1)) {
                // right
                r_vec_rank[j*n_curr + i] = \
                    F_i_j + \
                    2 / h1 * ksi_i_j;
            } else if ((curr_j == 1) && (curr_i < n - 1) && (curr_i > 0)) {
                // bottom
                r_vec_rank[j*n_curr + i] = \
                    F_i_j + \
                    1 / h2 / h2 * k(x_min + curr_i * h1, y_min + (curr_j - 0.5) * h2) * \
                    phi(x_min + curr_i * h1, y_min);
            } else if ((curr_j == m - 1) && (curr_i < n - 1) && (curr_i > 0)) {
                // top 
                r_vec_rank[j*n_curr + i] = \
                    F_i_j + \
                    2 / h2 * ksi_i_j;
            } else if ((curr_i == n - 1) && (curr_j == m - 1)) {
                // A2, B2
                r_vec_rank[j*n_curr + i] = \
                    F_i_j + \
                    (2 / h1 + 2 / h2) * ksi_i_j;
            } else if ((curr_i == n - 1) && (curr_j == 1)) {
                // A2, B1 + 1
                r_vec_rank[j*n_curr + i] = \
                    F_i_j + \
                    2 / h1 * ksi_i_j + \
                    1 / h2 / h2 * k(x_min + curr_i * h1, y_min + (curr_j - 0.5) * h2) * \
                    phi(x_min + curr_i * h1, y_min);
            } else if ((curr_i == 0) && (curr_j == m - 1)) {
                // A1, B2
                r_vec_rank[j*n_curr + i] = \
                    F_i_j + \
                    (2 / h1 + 2 / h2) * ksi_i_j;
            } else if ((curr_i == 0) && (curr_j == 1)) { 
                // A1, B1 + 1
                // w01
                r_vec_rank[j*n_curr + i] = \
                    F_i_j + \
                    2 / h1 * ksi_i_j + \
                    1 / h2 / h2 * k(x_min + curr_i * h1, y_min + (curr_j - 0.5) * h2) * \
                    phi(x_min + curr_i * h1, y_min + curr_j * h2);
            } else if (curr_j == 0) {
                // bot bottom
                r_vec_rank[j*n_curr + i] = 0;
            }  else {
                // inner
                r_vec_rank[j*n_curr + i] = F_i_j;
            }
        }
    }
    return 0;
}



int mpi_csr_mat_vec_mul(int n, int m, double *vals_rank, double *w_rank, 
        double *r_k, int *coords, int *dims_size, double *left_vector, 
        double *right_vector, double *bot_vector, double *top_vector)
{
    int curr_i, curr_j;
    int n_curr = n / dims_size[0];
    int m_curr = m / dims_size[1];
    #pragma omp parallel for collapse(2) num_threads(4)
    for (int j = 0; j < m_curr; j++) {
        for (int i = 0; i < n_curr; i++) {
            curr_i = coords[0] * n_curr + i;
            curr_j = coords[1] * m_curr + j;
            if ((curr_i == 0) && (curr_j != m - 1) && (curr_j != 0)) {
                // left
                r_k[j*n_curr + i] = 0.0;
                if (j == m_curr - 1) {
                    r_k[j*n_curr] += vals_rank[5*j*n_curr] * w_rank[j*n_curr - n_curr];
                    r_k[j*n_curr] += vals_rank[5*j*n_curr + 2] * w_rank[j*n_curr];
                    r_k[j*n_curr] += vals_rank[5*j*n_curr + 3] * w_rank[j*n_curr + 1];
                    r_k[j*n_curr] += vals_rank[5*j*n_curr + 4] * top_vector[curr_i];
                } else if (j == 0) {
                    r_k[j*n_curr] += vals_rank[5*j*n_curr] * bot_vector[curr_i];
                    r_k[j*n_curr] += vals_rank[5*j*n_curr + 2] * w_rank[j*n_curr];
                    r_k[j*n_curr] += vals_rank[5*j*n_curr + 3] * w_rank[j*n_curr + 1];
                    r_k[j*n_curr] += vals_rank[5*j*n_curr + 4] * w_rank[j*n_curr + n_curr];
                } else {
                    r_k[j*n_curr] += vals_rank[5*j*n_curr] * w_rank[j*n_curr - n_curr];
                    r_k[j*n_curr] += vals_rank[5*j*n_curr + 2] * w_rank[j*n_curr];
                    r_k[j*n_curr] += vals_rank[5*j*n_curr + 3] * w_rank[j*n_curr + 1];
                    r_k[j*n_curr] += vals_rank[5*j*n_curr + 4] * w_rank[j*n_curr + n_curr];
                }
            } else if ((curr_i == n - 1) && (curr_j != m - 1) && (curr_j != 0)) {
                // right
                r_k[j*n_curr + n_curr - 1] = 0.0;
                if (j == m_curr - 1) {
                    r_k[j*n_curr + n_curr - 1] += vals_rank[5*(j*n_curr + n_curr - 1)] * w_rank[j*n_curr + n_curr - 1 - n_curr];
                    r_k[j*n_curr + n_curr - 1] += vals_rank[5*(j*n_curr + n_curr - 1) + 1] * w_rank[j*n_curr + n_curr - 1 - 1];
                    r_k[j*n_curr + n_curr - 1] += vals_rank[5*(j*n_curr + n_curr - 1) + 2] * w_rank[j*n_curr + n_curr - 1];
                    r_k[j*n_curr + n_curr - 1] += vals_rank[5*(j*n_curr + n_curr - 1) + 4] * top_vector[n_curr - 1];
                } else if (j == 0) {
                    r_k[j*n_curr + n_curr - 1] += vals_rank[5*(j*n_curr + n_curr - 1)] * bot_vector[n_curr - 1];
                    r_k[j*n_curr + n_curr - 1] += vals_rank[5*(j*n_curr + n_curr - 1) + 1] * w_rank[j*n_curr + n_curr - 1 - 1];
                    r_k[j*n_curr + n_curr - 1] += vals_rank[5*(j*n_curr + n_curr - 1) + 2] * w_rank[j*n_curr + n_curr - 1];
                    r_k[j*n_curr + n_curr - 1] += vals_rank[5*(j*n_curr + n_curr - 1) + 4] * w_rank[j*n_curr + n_curr - 1 + n_curr];
                } else {
                    r_k[j*n_curr + n_curr - 1] += vals_rank[5*(j*n_curr + n_curr - 1)] * w_rank[j*n_curr + n_curr - 1 - n_curr];
                    r_k[j*n_curr + n_curr - 1] += vals_rank[5*(j*n_curr + n_curr - 1) + 1] * w_rank[j*n_curr + n_curr - 1 - 1];
                    r_k[j*n_curr + n_curr - 1] += vals_rank[5*(j*n_curr + n_curr - 1) + 2] * w_rank[j*n_curr + n_curr - 1];
                    r_k[j*n_curr + n_curr - 1] += vals_rank[5*(j*n_curr + n_curr - 1) + 4] * w_rank[j*n_curr + n_curr - 1 + n_curr];
                }
            } else if ((curr_j == m - 1) && (curr_i != n - 1) && (curr_i != 0)) {
                // top 
                r_k[(m_curr-1)*n_curr + i] = 0.0;
                if (i == 0) {
                    r_k[(m_curr-1)*n_curr + i] += vals_rank[5*(m_curr-1)*n_curr] * w_rank[(m_curr-1)*n_curr + i - n_curr];
                    r_k[(m_curr-1)*n_curr + i] += vals_rank[5*(m_curr-1)*n_curr + 1] * right_vector[m_curr - 1];
                    r_k[(m_curr-1)*n_curr + i] += vals_rank[5*(m_curr-1)*n_curr + 2] * w_rank[(m_curr-1)*n_curr + i];
                    r_k[(m_curr-1)*n_curr + i] += vals_rank[5*(m_curr-1)*n_curr + 3] * w_rank[(m_curr-1)*n_curr + i + 1];
                } else if (i == n_curr - 1) {
                    r_k[(m_curr-1)*n_curr + i] += vals_rank[5*(m_curr-1)*n_curr] * w_rank[(m_curr-1)*n_curr + i - n_curr];
                    r_k[(m_curr-1)*n_curr + i] += vals_rank[5*(m_curr-1)*n_curr + 1] * w_rank[(m_curr-1)*n_curr + i - 1];
                    r_k[(m_curr-1)*n_curr + i] += vals_rank[5*(m_curr-1)*n_curr + 2] * w_rank[(m_curr-1)*n_curr + i];
                    r_k[(m_curr-1)*n_curr + i] += vals_rank[5*(m_curr-1)*n_curr + 3] * right_vector[m_curr - 1];
                } else {
                    r_k[(m_curr-1)*n_curr + i] += vals_rank[5*(m_curr-1)*n_curr] * w_rank[(m_curr-1)*n_curr + i - n_curr];
                    r_k[(m_curr-1)*n_curr + i] += vals_rank[5*(m_curr-1)*n_curr + 1] * w_rank[(m_curr-1)*n_curr + i - 1];
                    r_k[(m_curr-1)*n_curr + i] += vals_rank[5*(m_curr-1)*n_curr + 2] * w_rank[(m_curr-1)*n_curr + i];
                    r_k[(m_curr-1)*n_curr + i] += vals_rank[5*(m_curr-1)*n_curr + 3] * w_rank[(m_curr-1)*n_curr + i + 1];
                }
            } else if ((curr_i == n - 1) && (curr_j == m - 1)) {
                // A2, B2
                r_k[m_curr*n_curr - 1] = 0.0;
                r_k[m_curr*n_curr - 1] += vals_rank[5*(m_curr*n_curr - 1)] * w_rank[(m_curr-1)*n_curr - n_curr];
                r_k[m_curr*n_curr - 1] += vals_rank[5*(m_curr*n_curr - 1) + 1] * w_rank[(m_curr-1)*n_curr - 1];
                r_k[m_curr*n_curr - 1] += vals_rank[5*(m_curr*n_curr - 1) + 2] * w_rank[(m_curr-1)*n_curr];
            // } else if ((curr_i == n - 1) && (curr_j == 1)) {
            //     // A2, B1 + 1
            } else if ((curr_i == 0) && (curr_j == m - 1)) {
                // A1, B2
                r_k[(m_curr-1)*n_curr] = 0.0;
                r_k[(m_curr-1)*n_curr] += vals_rank[5*(m_curr-1)*n_curr] * w_rank[(m_curr-1)*n_curr - n_curr];
                r_k[(m_curr-1)*n_curr] += vals_rank[5*(m_curr-1)*n_curr + 2] * w_rank[(m_curr-1)*n_curr];
                r_k[(m_curr-1)*n_curr] += vals_rank[5*(m_curr-1)*n_curr + 3] * w_rank[(m_curr-1)*n_curr + 1];
            // } else if ((curr_i == 0) && (curr_j == 1)) { 
            //     // A1, B1 + 1
            } else if (curr_j == 0) {
                // bot bottom
                r_k[j*n_curr + i] = 0;
            } else {
                // inner
                r_k[j*n_curr + i] = 0.0;
                if ((i == n_curr - 1) && (j == m_curr - 1)) {
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i)] * w_rank[j*n_curr + i - n_curr];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 1] * w_rank[j*n_curr + i - 1];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 2] * w_rank[j*n_curr + i];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 3] * right_vector[j];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 4] * top_vector[i];
                } else if ((i == n_curr - 1) && (j == 0)) {
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i)] * bot_vector[i];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 1] * w_rank[j*n_curr + i - 1];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 2] * w_rank[j*n_curr + i];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 3] * right_vector[j];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 4] * w_rank[j*n_curr + i + n_curr];
                } else if ((i == 0) && (j == m_curr - 1)) {
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i)] * w_rank[j*n_curr + i - n_curr];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 1] * left_vector[j];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 2] * w_rank[j*n_curr + i];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 3] * w_rank[j*n_curr + i + 1];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 4] * top_vector[i];
                } else if ((i == 0) && (j == 0)) {
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i)] * bot_vector[i];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 1] * left_vector[j];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 2] * w_rank[j*n_curr + i];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 3] * w_rank[j*n_curr + i + 1];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 4] * w_rank[j*n_curr + i + n_curr];
                } else if (i == n_curr - 1) {
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i)] * w_rank[j*n_curr + i - n_curr];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 1] * w_rank[j*n_curr + i - 1];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 2] * w_rank[j*n_curr + i];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 3] * right_vector[j];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 4] * w_rank[j*n_curr + i + n_curr];
                } else if (i == 0) {
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i)] * w_rank[j*n_curr + i - n_curr];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 1] * left_vector[j];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 2] * w_rank[j*n_curr + i];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 3] * w_rank[j*n_curr + i + 1];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 4] * w_rank[j*n_curr + i + n_curr];
                } else if (j == m_curr - 1) {
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i)] * w_rank[j*n_curr + i - n_curr];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 1] * w_rank[j*n_curr + i - 1];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 2] * w_rank[j*n_curr + i];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 3] * w_rank[j*n_curr + i + 1];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 4] * top_vector[i];
                } else if (j == 0) {
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i)] * bot_vector[i];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 1] * w_rank[j*n_curr + i - 1];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 2] * w_rank[j*n_curr + i];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 3] * w_rank[j*n_curr + i + 1];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 4] * w_rank[j*n_curr + i + n_curr];
                } else {
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i)] * w_rank[j*n_curr + i - n_curr];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 1] * w_rank[j*n_curr + i - 1];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 2] * w_rank[j*n_curr + i];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 3] * w_rank[j*n_curr + i + 1];
                    r_k[j*n_curr + i] += vals_rank[5*(j*n_curr + i) + 4] * w_rank[j*n_curr + i + n_curr];
                }
            }
        }
    }
    return 0;
}

int mat_vec_mul_sub(int n, int m, double *matx, double *w, double *r_vec, double *r_k) {
    for (int i = 0; i < n*m; i++) { 
        // for (int k = 0; k < n*m; k++) { 
            r_k[i] = 0.0;
            for (int j = 0; j < n*m; j++) { 
                r_k[i] += matx[i*n*m + j] * w[j];
                // r_vec[i] += matx[i*n*m + j] * w[k];
            }
            r_k[i] -= r_vec[i];
        // }
    }
    return 0;
}

int mat_vec_mul(int n, int m, double *matx, double *w, double *res) {
    for (int i = 0; i < n*m; i++) { 
        // for (int k = 0; k < n*m; k++) {
            res[i] = 0.0;
            for (int j = 0; j < n*m; j++) { 
                res[i] += matx[i*n*m + j] * w[j];  
                // r_vec[i] += matx[i*n*m + j] * w[k];
            }
        // }
    }
    return 0;
}

int vecsub(int n, int m, double *A, double *B, double *res) {
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            res[j*n + i] = A[j*n + i] - B[j*n + i];
            // if ((i + 1) % n == 0) {
            //     std::cout << res[i] << std::endl;
            // }
        }
    }
    return 0;
}


double mpi_norm_e_matx_sqr(int n, int m, double h1, double h2, int *coords, int *dims_size, double *u, double *v)
{
    double sum = 0;
    double ro_1;
    double ro_2;
    int curr_i, curr_j;
    int n_curr = n / dims_size[0];
    int m_curr = m / dims_size[1];
    for (int j = 0; j < m_curr; j++) {
        for (int i = 0; i < n_curr; i++) {
            curr_i = coords[0] * n_curr + i;
            curr_j = coords[1] * m_curr + j;
            ro_1 = 1;
            ro_2 = 1;
            if ((curr_i == 0) || (curr_i == n - 1)) {
                ro_1 = 0.5;
            } 
            if ((curr_j == 0) || (curr_j == m - 1)) {
                ro_2 = 0.5;
            }
            sum += h1 * h2 * ro_1 * ro_2 * u[j*n_curr+i] * v[j*n_curr+i];
        }
    }
    return sum;
}

double vec_scalar(int n, int m, double *A, double *B) {
    double sum = 0;
    for (int i = 0; i < n*m; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}

double norm_e_sqr(int n, int m, double *vec) {
    double sum = 0.0;
    for (int i = 0; i < n*m; i++) {
        sum += vec[i] * vec[i];
    }
    return sum;
}

int rebalance(int n, int m, double *w, double tau_k, double *r_k) {
    for (int i = n; i < n*m; i++) {
        r_k[i] *= tau_k; // for norm
        w[i] -= r_k[i];
    }
    return 0;
}


double mpi_computing(int n, int m, double h1, double h2, double *vals_rank, double *w_rank, double *r_vec_rank, int *coords, int *dims_size, MPI_Comm *comm) {
    int n_curr = n / dims_size[0];
    int m_curr = m / dims_size[1];
    double eps = 1.0e-6;
    double err_each, err = 2 * eps;
    double *r_k = new double[n_curr * m_curr];
    double *Ar = new double[n_curr * m_curr];
    double *bot_vector = new double[n_curr];
    double *top_vector = new double[n_curr];
    double *left_vector = new double[m_curr];
    double *right_vector = new double[m_curr];
    double tau_k = 0.0;
    double ar_norm_rk_each[2] = {0.0, 0.0};
    double ar_norm_rk[2] = {0.0, 0.0};
    // double ar_norm_each, ar_rk_each, ar_rk, ar_norm;
    MPI_Request request_right, request_left, request_bot, request_top;
    MPI_Status status_right, status_left, status_bot, status_top;
    MPI_Datatype row_type, column_type;
    MPI_Type_vector(1, n / dims_size[0], 1, MPI_DOUBLE, &row_type);
    MPI_Type_vector(m / dims_size[1], 1, n / dims_size[0], MPI_DOUBLE, &column_type);
    MPI_Type_commit(&row_type);
    MPI_Type_commit(&column_type);
    int rank;
    MPI_Cart_rank(*comm, coords, &rank);
    int rank_left = rank, rank_right = rank, rank_top = rank, rank_bot = rank;
    if (coords[0] != 0) {
        coords[0] -= 1;
        MPI_Cart_rank(*comm, coords, &rank_left);
        coords[0] += 1;
    }
    if (coords[0] != dims_size[0] - 1) {
        coords[0] += 1;
        MPI_Cart_rank(*comm, coords, &rank_right);
        coords[0] -= 1;
    }
    if (coords[1] != 0) {
        coords[1] -= 1;
        MPI_Cart_rank(*comm, coords, &rank_bot);
        coords[1] += 1;
    }
    if (coords[1] != dims_size[1] - 1) {
        coords[1] += 1;
        MPI_Cart_rank(*comm, coords, &rank_top);
        coords[1] -= 1;
    }
    int i = 0;
    while (err > eps) {
        // vertical exchange
        if ((coords[1] == 0) || (coords[1] == dims_size[1] - 1)) {
            if (coords[1] == 0) {
                MPI_Isend(&w_rank[(m_curr - 1)*n_curr], 1, row_type,
                        rank_top, rank, *comm, &request_top);
                MPI_Recv(top_vector, n_curr, MPI_DOUBLE,
                        rank_top, rank_top, *comm, &status_top);
            }

            if (coords[1] == dims_size[1] - 1) {
                MPI_Isend(w_rank, 1, row_type,
                        rank_bot, rank, *comm, &request_bot);
                MPI_Recv(bot_vector, n_curr, MPI_DOUBLE,
                        rank_bot, rank_bot, *comm, &status_bot);
            }
        } else {
            MPI_Isend(w_rank, 1, row_type,
                    rank_bot, rank, *comm, &request_bot);
            MPI_Isend(&w_rank[(m_curr - 1)*n_curr], 1, row_type,
                    rank_top, rank, *comm, &request_top);
            
            MPI_Recv(bot_vector, n_curr, MPI_DOUBLE,
                    rank_bot, rank_bot, *comm, &status_bot);
            MPI_Recv(top_vector, n_curr, MPI_DOUBLE,
                    rank_top, rank_top, *comm, &status_top);
        }

        // horizontal exchange
        if ((coords[0] == 0) || (coords[0] == dims_size[0] - 1)) {
            if (coords[0] == 0) {
                MPI_Isend(&w_rank[n_curr-1], 1, column_type,
                        rank_right, rank, *comm, &request_right);
                MPI_Recv(right_vector, m_curr, MPI_DOUBLE,
                        rank_right, rank_right, *comm, &status_right);
            }

            if (coords[0] == dims_size[0] - 1) {
                MPI_Isend(w_rank, 1, column_type,
                        rank_left, rank, *comm, &request_left);
                MPI_Recv(left_vector, m_curr, MPI_DOUBLE,
                        rank_left, rank_left, *comm, &status_left);
            }
        } else {
            MPI_Isend(w_rank, 1, column_type,
                    rank_left, rank, *comm, &request_left);
            MPI_Isend(&w_rank[n_curr-1], 1, column_type,
                    rank_right, rank, *comm, &request_right);

            MPI_Recv(left_vector, m_curr, MPI_DOUBLE,
                    rank_left, rank_left, *comm, &status_left);
            MPI_Recv(right_vector, m_curr, MPI_DOUBLE,
                    rank_right, rank_right, *comm, &status_right);
        }
        // mpi_matx_print(n/dims_size[0], m/dims_size[1], right_vector, coords, dims_size);

        mpi_csr_mat_vec_mul(n, m, vals_rank, w_rank, r_k, coords, dims_size,
                left_vector, right_vector, bot_vector, top_vector); // r_k = Aw 
        vecsub(n/dims_size[0], m/dims_size[1], r_k, r_vec_rank, r_k); // r_k -= B or r_k = Aw - B
        // mpi_matx_print(n/dims_size[0], m/dims_size[1], r_k, coords, dims_size);
        
        
        // vertical exchange
        if ((coords[1] == 0) || (coords[1] == dims_size[1] - 1)) {
            if (coords[1] == 0) {
                MPI_Isend(&r_k[(m_curr - 1)*n_curr], 1, row_type,
                        rank_top, rank, *comm, &request_top);
                MPI_Recv(top_vector, n_curr, MPI_DOUBLE,
                        rank_top, rank_top, *comm, &status_top);
            }

            if (coords[1] == dims_size[1] - 1) {
                MPI_Isend(r_k, 1, row_type,
                        rank_bot, rank, *comm, &request_bot);
                MPI_Recv(bot_vector, n_curr, MPI_DOUBLE,
                        rank_bot, rank_bot, *comm, &status_bot);
            }
        } else {
            MPI_Isend(r_k, 1, row_type,
                    rank_bot, rank, *comm, &request_bot);
            MPI_Isend(&r_k[(m_curr - 1)*n_curr], 1, row_type,
                    rank_top, rank, *comm, &request_top);
            
            MPI_Recv(bot_vector, n_curr, MPI_DOUBLE,
                    rank_bot, rank_bot, *comm, &status_bot);
            MPI_Recv(top_vector, n_curr, MPI_DOUBLE,
                    rank_top, rank_top, *comm, &status_top);
        }

        // horizontal exchange
        if ((coords[0] == 0) || (coords[0] == dims_size[0] - 1)) {
            if (coords[0] == 0) {
                MPI_Isend(&r_k[n_curr-1], 1, column_type,
                        rank_right, rank, *comm, &request_right);
                MPI_Recv(right_vector, m_curr, MPI_DOUBLE,
                        rank_right, rank_right, *comm, &status_right);
            }

            if (coords[0] == dims_size[0] - 1) {
                MPI_Isend(r_k, 1, column_type,
                        rank_left, rank, *comm, &request_left);
                MPI_Recv(left_vector, m_curr, MPI_DOUBLE,
                        rank_left, rank_left, *comm, &status_left);
            }
        } else {
            MPI_Isend(r_k, 1, column_type,
                    rank_left, rank, *comm, &request_left);
            MPI_Isend(&r_k[n_curr-1], 1, column_type,
                    rank_right, rank, *comm, &request_right);

            MPI_Recv(left_vector, m_curr, MPI_DOUBLE,
                    rank_left, rank_left, *comm, &status_left);
            MPI_Recv(right_vector, m_curr, MPI_DOUBLE,
                    rank_right, rank_right, *comm, &status_right);
        }
        mpi_csr_mat_vec_mul(n, m, vals_rank, r_k, Ar, coords, dims_size, 
                left_vector, right_vector, bot_vector, top_vector); // Ar = Ar_k
        // mpi_matx_print(n/dims_size[0], m/dims_size[1], Ar, coords, dims_size);
        
        
        ar_norm_rk_each[0] = mpi_norm_e_matx_sqr(n, m, h1, h2, coords, dims_size, Ar, r_k);
        ar_norm_rk_each[1] = mpi_norm_e_matx_sqr(n, m, h1, h2, coords, dims_size, Ar, Ar);
        // std::cout << rank << " " << ar_norm_rk_each[0] << " " << ar_norm_rk_each[1] << std::endl;
        // ar_rk = 0.0;
        // ar_norm = 0.0;
        ar_norm_rk[0] = 0.0;
        ar_norm_rk[1] = 0.0;
        MPI_Allreduce(&ar_norm_rk_each, &ar_norm_rk, 2, MPI_DOUBLE, MPI_SUM, *comm);
        tau_k = ar_norm_rk[0] / ar_norm_rk[1];
        rebalance(n/dims_size[0], m/dims_size[1], w_rank, tau_k, r_k); // w_k+1 = w_k - tau_k+1 * r_k
        
        err_each = mpi_norm_e_matx_sqr(n, m, h1, h2, coords, dims_size, r_k, r_k);
        err = 0.0;
        MPI_Allreduce(&err_each, &err, 1, MPI_DOUBLE, MPI_SUM, *comm);
        err = sqrt(err);
        // if (rank == 0) {
        //     std::cout << "iter" << i << " " << err << " " << ar_norm_rk[0] << " " << ar_norm_rk[1] << std::endl;
        //     i += 1;
        // }
        // err = eps / 2;
    }
    
    delete[] Ar;
    delete[] r_k;
    delete[] bot_vector;
    delete[] top_vector;
    delete[] left_vector;
    delete[] right_vector;
    return err;
}

