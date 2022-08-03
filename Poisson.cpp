/*
    Solve the Poisson problem
        u_{xx} + u_{yy} = f(x, y)   x \in \Omega = [0, pi] x [0, pi]
    with
        u(0, y) = u(pi, y) = 0
        u(x, 0) = 2 sin x
        u(x, pi) = -2 sin x
    and
        f(x, y) = -20 sin x cos 3 y
    using Jacobi iterations and MPI. A grided
division of work is implemented and the number of processes being requested is a square number.
*/

// MPI Library
#include "mpi.h"

// Standard IO libraries
#include <iostream>
#include <fstream>
using namespace std;

#include <math.h>

int main(int argc, char *argv[])
{

    // Problem paramters
    double const pi = 3.141592654;
    double const a = 0.0, b = pi;

    // Numerical parameters
    int const MAX_ITERATIONS = pow(2, 16), PRINT_INTERVAL = 1000;
    int N, k;
    double x, y, dx, dy, tolerance, du_max;

    // MPI Variables
    int num_procs, rank, tag, rank_N, x_pos, y_pos, sqrt_num_procs;
    double du_max_proc;
    MPI_Status status;
    MPI_Request request;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // TODO: Handle the case of only one process
    if (num_procs == 1)
    {
        cout << "This program only handles more than one process.\n";
        MPI_Finalize();
        return 0;
    }

    // Discretization
    N = 100;
    dx = (pi - 0) / ((double)(N + 1));
    dy = dx;
    tolerance = 0.1 * pow(dx, 2);

    // label process position
    sqrt_num_procs = (int)(sqrt(num_procs));
    x_pos = rank % sqrt_num_procs;
    y_pos = rank / sqrt_num_procs;

    // Organization of local process (rank) data
    rank_N = (N + sqrt_num_procs - 1) / sqrt_num_procs;

    cout << "Rank: " << rank << " x_pos: " << x_pos << " y_pos: " << y_pos<< " \n";


    // Allocate work arrays
    double *send_buffer = new double[rank_N];
    double *recv_buffer = new double[rank_N];
    double **u = new double*[rank_N + 2];
    double **u_old = new double*[rank_N + 2];
    double **f = new double*[rank_N+ 2];
    for (int i = 0; i < rank_N + 2; ++i)
    {
        u[i] = new double[rank_N + 2];
        u_old[i] = new double[rank_N + 2];
        f[i] = new double[rank_N + 2];
    }

    // For reference, (x_i, y_j) u[i][j]
    // so that i references columns and j rows
    // Initialize arrays - fill boundaries
    for (int i = 0; i < rank_N + 2; ++i)
    {
        x = dx * (double) i + a + x_pos* (rank_N) * dx;
        for (int j = 0; j < rank_N + 2; ++j)
        {
            y = dy * (double) (j + y_pos* (rank_N)) + a;
            f[i][j] = -20.0 * sin(x) * cos(3.0 * y);
            u[i][j] = 1.0;
        }
    }

    // Set boundaries
    // Set bottom
    if (y_pos == 0)
    {
        for (int i = 0; i < rank_N + 2; ++i)
        {
            x = dx * (double) i + a + x_pos* (rank_N) * dx;
            u[i][0] = 2.0 * sin(x);
        }
    }
    // Set top
    if (y_pos == sqrt_num_procs - 1)
    {
        for (int i = 0; i < rank_N + 2; ++i)
        {
            x = dx * (double) i + a + x_pos* (rank_N) * dx;
            u[i][rank_N + 1] = -2.0 * sin(x);
        }
    }
    // Set left
    if (x_pos == 0)
    {
      for (int j = 0; j < rank_N + 2; ++j)
      {
          u[0][j] = 0.0;
      }
    }
    // Set right
    if (x_pos == sqrt_num_procs - 1)
    {
      for (int j = 0; j < rank_N + 2; ++j)
      {
          u[rank_N + 1][j] = 0.0;
      }
    }




    // Inital copy into u_old - note that this does not require communication
    // as we know all values on each process at this point
    for (int i = 0; i < rank_N + 2; ++i)
        for (int j = 0; j < rank_N + 2; ++j)
            u_old[i][j] = u[i][j];

    /* Jacobi Iterations */
    k = 0;
    while (k < MAX_ITERATIONS)
    {
        k++;

        du_max_proc = 0.0;
        for (int i = 1; i < rank_N + 1; ++i)
        {
            for (int j = 1; j < rank_N + 1; ++j)
            {
                u[i][j] = 0.25 * (u_old[i-1][j] + u_old[i+1][j] + u_old[i][j-1] + u_old[i][j+1] - pow(dx, 2) * f[i][j]);
                du_max_proc = fmax(du_max_proc, fabs(u[i][j] - u_old[i][j]));
            }
        }

        // Final global max change in solution
        MPI_Allreduce(&du_max_proc, &du_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0)
            if (k%PRINT_INTERVAL == 0)
                cout << "After " << k << " iterations, du_max = " << du_max << "\n";

        if (du_max < tolerance)
            break;

        // Copy into old data that we have
        for (int i = 1; i < rank_N + 1; ++i)
            for (int j = 1; j < rank_N + 1; ++j)
                u_old[i][j] = u[i][j];

        // Communicate data
        // Send data up (tag = 1)
        if (y_pos < sqrt_num_procs - 1)
        {
            for (int i = 1; i < rank_N + 1; ++i)
                send_buffer[i-1] = u_old[i][rank_N];
            MPI_Isend(send_buffer, rank_N, MPI_DOUBLE_PRECISION, rank + sqrt_num_procs, 1, MPI_COMM_WORLD, &request);
        }
        // Send data down (tag = 2)
        if (y_pos > 0)
        {
            for (int i = 1; i < rank_N + 1; ++i)
                send_buffer[i - 1] = u_old[i][1];
            MPI_Isend(send_buffer, rank_N, MPI_DOUBLE_PRECISION, rank - sqrt_num_procs, 2, MPI_COMM_WORLD, &request);
        }

        // Receive data from above (tag = 2)
        if (y_pos < sqrt_num_procs - 1)
        {
            MPI_Recv(recv_buffer, rank_N, MPI_DOUBLE_PRECISION, rank + sqrt_num_procs, 2, MPI_COMM_WORLD, &status);
            for (int i = 1; i < rank_N + 1; ++i)
                u_old[i][rank_N + 1] = recv_buffer[i - 1];
        }

        // Receive data from below (tag = 1)
        if (y_pos > 0)
        {
            MPI_Recv(recv_buffer, rank_N, MPI_DOUBLE_PRECISION, rank - sqrt_num_procs, 1, MPI_COMM_WORLD, &status);
            for (int i = 1; i < rank_N + 1; ++i)
                u_old[i][0] = recv_buffer[i - 1];
        }

        // Communicate data
        // Send data right(tag = 3)
        if (x_pos < sqrt_num_procs - 1)
        {
            for (int i = 1; i < rank_N + 1; ++i)
                send_buffer[i-1] = u_old[rank_N][i];
            MPI_Isend(send_buffer, rank_N, MPI_DOUBLE_PRECISION, rank + 1, 3, MPI_COMM_WORLD, &request);
        }
        // Send data left (tag = 4)
        if (x_pos > 0)
        {
            for (int i = 1; i < rank_N + 1; ++i)
                send_buffer[i - 1] = u_old[1][i];
            MPI_Isend(send_buffer, rank_N, MPI_DOUBLE_PRECISION, rank - 1, 4, MPI_COMM_WORLD, &request);
        }

        // Receive data from right (tag = 4)
        if (x_pos < sqrt_num_procs - 1)
        {
            MPI_Recv(recv_buffer, rank_N, MPI_DOUBLE_PRECISION, rank + 1, 4, MPI_COMM_WORLD, &status);
            for (int i = 1; i < rank_N + 1; ++i)
                u_old[rank_N + 1][i] = recv_buffer[i - 1];
        }

        // Receive data from left (tag = 3)
        if (x_pos > 0)
        {
            MPI_Recv(recv_buffer, rank_N, MPI_DOUBLE_PRECISION, rank - 1, 3, MPI_COMM_WORLD, &status);
            for (int i = 1; i < rank_N + 1; ++i)
                u_old[0][i] = recv_buffer[i - 1];
        }


    }

    // Output Results
    // Check for failure
    if (N >= MAX_ITERATIONS)
    {

        if (rank == 0)
        {
            cout << "*** Jacobi failed to converge!\n";
            cout << "***   Reached du_max = " << du_max << "\n";
            cout << "***   Tolerance = " << tolerance << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    string file_name = "jacobi_" + to_string(rank) + ".txt";
    ofstream fp(file_name);

    // Write to files, rank_N x rank_N matrix for each file
    for (int j = 1; j < rank_N + 1; ++j)
    {
        for (int i = 1; i < rank_N + 1; ++i)
            fp << u[i][j] << " ";
        fp << "\n";
    }


    fp.close();

    // Check numbers
    x = dx * (double) (rank_N) + a + x_pos* (rank_N) * dx;
    y = dy * (double) (rank_N + y_pos* (rank_N)) + a;
    cout << "x,y: " << x <<"," << y << "\n";
    cout << "True Sol - Numeric Sol: " << 2 * sin(x) * cos(3 * y) - u[rank_N][rank_N] << "\n";
    //cout << "True Solution: " << 2 * sin(x) * cos(3 * y) << "Numerical Solution: " << u[rank_N][rank_N] << "\n";


    MPI_Finalize();

    return 0;
}
