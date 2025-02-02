#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Simulation constants (diffusion parameters)
const float D = 0.1f;        // Diffusion constant.
const float DELTA_T = 0.01f; // Time step.
const float DELTA_X = 1.0f;  // Spatial resolution.

// Macro to index into a 1D array representing a 2D array.
#define IDX(i, j, ncols) ((i) * (ncols) + (j))

// Helper function to check whether a given global (i,j) is a boundary cell.
bool is_global_boundary(int i, int j, int global_n)
{
    return (i == 0 || i == global_n - 1 || j == 0 || j == global_n - 1);
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    FILE *fp = NULL;
    // Only rank 0 opens the output file.
    if (rank == 0)
    {
        fp = fopen("results.csv", "w");
        if (fp == NULL)
        {
            fprintf(stderr, "Error opening results.csv for writing.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Write CSV header.
        fprintf(fp, "try, matrix_size, iterations, setup_time, runtime_time, teardown_time, mean_value\n");
        fflush(fp);
    }

    /*
     * Loop over the different matrix sizes and iteration counts.
     * Matrix sizes: 2000, 4000, 6000 (square matrix: global_n x global_n)
     * Iteration counts: 500, 1000, 1500
     * Each combination is run 3 times.
     */
    for (int global_n = 2000; global_n <= 6000; global_n += 2000)
    {
        for (int iterations = 500; iterations <= 1500; iterations += 500)
        {
            for (int trial = 1; trial <= 3; trial++)
            {

                double t_setup_start, t_setup_end;
                double t_runtime_start, t_runtime_end;
                double t_teardown_start, t_teardown_end;

                /* ===== Setup Phase ===== */
                t_setup_start = MPI_Wtime();

                // Partition the rows among processes.
                int base_rows = global_n / size;
                int remainder = global_n % size;
                int local_rows = base_rows + (rank < remainder ? 1 : 0);

                // Determine the global row index of the first row for this process.
                int row_start = rank * base_rows + (rank < remainder ? rank : remainder);

                // Allocate arrays for the local block plus two ghost rows.
                int total_rows = local_rows + 2; // ghost top and bottom rows
                float *current = (float *)malloc(total_rows * global_n * sizeof(float));
                float *next = (float *)malloc(total_rows * global_n * sizeof(float));
                if (!current || !next)
                {
                    fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                // Initialize the interior (real) rows to 0.
                for (int i = 1; i <= local_rows; i++)
                {
                    for (int j = 0; j < global_n; j++)
                    {
                        current[IDX(i, j, global_n)] = 0.0f;
                    }
                }
                // Initialize ghost rows to 0.
                for (int j = 0; j < global_n; j++)
                {
                    current[IDX(0, j, global_n)] = 0.0f;
                    current[IDX(local_rows + 1, j, global_n)] = 0.0f;
                }

                // Set the center cell of the global matrix to 1.
                int center = global_n / 2;
                if (center >= row_start && center < row_start + local_rows)
                {
                    int local_center = center - row_start + 1; // +1 to account for ghost row.
                    current[IDX(local_center, center, global_n)] = 1.0f;
                }

                // Copy current into next for the initial condition.
                for (int i = 0; i < total_rows; i++)
                {
                    for (int j = 0; j < global_n; j++)
                    {
                        next[IDX(i, j, global_n)] = current[IDX(i, j, global_n)];
                    }
                }
                t_setup_end = MPI_Wtime();

                /* ===== Runtime Phase (Iterations) ===== */
                t_runtime_start = MPI_Wtime();

                for (int iter = 0; iter < iterations; iter++)
                {
                    MPI_Request reqs[4];
                    int req_count = 0;

                    // Exchange halo rows with neighbors.
                    if (rank > 0)
                    {
                        MPI_Isend(&current[IDX(1, 0, global_n)], global_n, MPI_FLOAT,
                                  rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
                        MPI_Irecv(&current[IDX(0, 0, global_n)], global_n, MPI_FLOAT,
                                  rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
                    }
                    if (rank < size - 1)
                    {
                        MPI_Isend(&current[IDX(local_rows, 0, global_n)], global_n, MPI_FLOAT,
                                  rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
                        MPI_Irecv(&current[IDX(local_rows + 1, 0, global_n)], global_n, MPI_FLOAT,
                                  rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
                    }
                    if (req_count > 0)
                        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

                    // Update the interior cells.
                    for (int i = 1; i <= local_rows; i++)
                    {
                        int global_i = row_start + (i - 1);
                        for (int j = 0; j < global_n; j++)
                        {
                            if (is_global_boundary(global_i, j, global_n))
                            {
                                next[IDX(i, j, global_n)] = current[IDX(i, j, global_n)];
                            }
                            else
                            {
                                float laplacian = (current[IDX(i + 1, j, global_n)] + current[IDX(i - 1, j, global_n)] + current[IDX(i, j + 1, global_n)] + current[IDX(i, j - 1, global_n)] - 4.0f * current[IDX(i, j, global_n)]) / (DELTA_X * DELTA_X);
                                next[IDX(i, j, global_n)] = current[IDX(i, j, global_n)] + D * DELTA_T * laplacian;
                            }
                        }
                    }

                    // Swap the buffers.
                    float *tmp = current;
                    current = next;
                    next = tmp;
                }
                t_runtime_end = MPI_Wtime();

                /* ===== Teardown Phase ===== */
                t_teardown_start = MPI_Wtime();
                double local_sum = 0.0;
                for (int i = 1; i <= local_rows; i++)
                {
                    for (int j = 0; j < global_n; j++)
                    {
                        local_sum += current[IDX(i, j, global_n)];
                    }
                }
                double global_sum = 0.0;
                MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                double mean = global_sum / (global_n * (double)global_n);
                t_teardown_end = MPI_Wtime();

                // Free allocated memory.
                free(current);
                free(next);

                /* ===== Print and Save Measurements ===== */
                double setup_time = t_setup_end - t_setup_start;
                double runtime_time = t_runtime_end - t_runtime_start;
                double teardown_time = t_teardown_end - t_teardown_start;

                // Only rank 0 writes to the CSV file.
                if (rank == 0)
                {
                    fprintf(fp, "%d, %d, %d, %lf, %lf, %lf, %lf\n",
                            trial, global_n, iterations,
                            setup_time, runtime_time, teardown_time, mean);
                    fflush(fp);
                }
            } // End trial loop.
        } // End iterations loop.
    } // End matrix size loop.

    if (rank == 0)
    {
        fclose(fp);
    }

    MPI_Finalize();
    return 0;
}
