#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define START_ITERATIONS 500
#define MAX_ITERATIONS 1500
#define ITERATIONS_STEP 500

#define MAX_TRIES 3

#define START_GRID_SIZE 2000
#define MAX_GRID_SIZE 6000
#define GRID_SIZE_STEP 2000

#define DELTA_T 0.01
#define DELTA_X 1.0
#define DIFFUSION_COEFFICIENT 0.1

#define IDX(row, column, height) ((row) * (height) + (column))

bool is_global_boundary(int i, int j, int global_n)
{
    return (i == 0 || i == global_n - 1 || j == 0 || j == global_n - 1);
}

int is_global_center(int grid_size, int number_of_processes, int my_rank)
{
    int base_rows = grid_size / number_of_processes,
        remainder = grid_size % number_of_processes,
        local_rows = base_rows + (my_rank < remainder ? 1 : 0),
        center = grid_size / 2,
        row_start = my_rank * base_rows + (my_rank < remainder ? my_rank : remainder);

    return center >= row_start && center < row_start + local_rows;
}

int run_simulation(int grid_size, int iterations, int my_rank, int number_of_processes, double (*reader)[4])
{
    double end_setup_time, start_setup_time, t_runtime_end, t_runtime_start, t_teardown_end, t_teardown_start;
    float *temporary_grid, *next_grid, *current_grid;

    start_setup_time = MPI_Wtime();

    int base_rows = grid_size / number_of_processes,
        remainder = grid_size % number_of_processes,
        local_rows = base_rows + (my_rank < remainder ? 1 : 0),
        center = grid_size / 2,
        row_start = my_rank * base_rows + (my_rank < remainder ? my_rank : remainder),
        total_rows = local_rows + 2;

    next_grid = (float *)malloc(total_rows * grid_size * sizeof(float));
    current_grid = (float *)malloc(total_rows * grid_size * sizeof(float));

    if (!current_grid || !next_grid)
    {
        fprintf(stderr, "Process %d: Memory allocation failed\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 1; i <= local_rows; i++)
        for (int j = 0; j < grid_size; j++)
            current_grid[IDX(i, j, grid_size)] = 0.0f;

    for (int j = 0; j < grid_size; j++)
    {
        current_grid[IDX(0, j, grid_size)] = 0.0f;
        current_grid[IDX(local_rows + 1, j, grid_size)] = 0.0f;
    }

    if (is_global_center(grid_size, number_of_processes, my_rank))
    {
        int local_center = center - row_start + 1;

        current_grid[IDX(local_center, center, grid_size)] = 1.0f;
    }

    end_setup_time = MPI_Wtime();

    t_runtime_start = MPI_Wtime();

    for (int iteration = 0; iteration < iterations; iteration++)
    {
        MPI_Request requests[4];

        int request_count = 0;

        if (my_rank > 0)
        {
            MPI_Isend(&current_grid[IDX(1, 0, grid_size)], grid_size, MPI_FLOAT, my_rank - 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
            MPI_Irecv(&current_grid[IDX(0, 0, grid_size)], grid_size, MPI_FLOAT, my_rank - 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
        }

        if (my_rank < number_of_processes - 1)
        {
            MPI_Isend(&current_grid[IDX(local_rows, 0, grid_size)], grid_size, MPI_FLOAT, my_rank + 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
            MPI_Irecv(&current_grid[IDX(local_rows + 1, 0, grid_size)], grid_size, MPI_FLOAT, my_rank + 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
        }

        if (request_count > 0)
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

        for (int i = 1; i <= local_rows; i++)
        {
            int global_i = row_start + (i - 1);

            for (int j = 0; j < grid_size; j++)
                if (is_global_boundary(global_i, j, grid_size))
                    next_grid[IDX(i, j, grid_size)] = current_grid[IDX(i, j, grid_size)];
                else
                {
                    float laplacian = (current_grid[IDX(i + 1, j, grid_size)] + current_grid[IDX(i - 1, j, grid_size)] + current_grid[IDX(i, j + 1, grid_size)] + current_grid[IDX(i, j - 1, grid_size)] - 4.0f * current_grid[IDX(i, j, grid_size)]) / (DELTA_X * DELTA_X);

                    next_grid[IDX(i, j, grid_size)] = current_grid[IDX(i, j, grid_size)] + DIFFUSION_COEFFICIENT * DELTA_T * laplacian;
                }
        }

        float *temporary_grid = current_grid;

        current_grid = next_grid;

        next_grid = temporary_grid;
    }

    t_runtime_end = MPI_Wtime();

    t_teardown_start = MPI_Wtime();

    if (center >= row_start && center < row_start + local_rows)
    {
        int local_center = center - row_start + 1;
        (*reader)[0] = next_grid[IDX(local_center, center, grid_size)];
    }
    else
        (*reader)[0] = 0.0f;

    free(current_grid);
    free(next_grid);

    t_teardown_end = MPI_Wtime();

    (*reader)[1] = end_setup_time - start_setup_time;
    (*reader)[2] = t_runtime_end - t_runtime_start;
    (*reader)[3] = t_teardown_end - t_teardown_start;
}

int main(int argc, char *argv[])
{
    int my_rank, number_of_processes;

    double reader[4];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

    FILE *file = NULL;

    file = fopen("results.csv", "a+");

    if (file == NULL)
    {
        fprintf(stderr, "Error opening results.csv for writing.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int grid_size = START_GRID_SIZE; grid_size <= MAX_GRID_SIZE; grid_size += GRID_SIZE_STEP)
        for (int iterations = START_ITERATIONS; iterations <= MAX_ITERATIONS; iterations += ITERATIONS_STEP)
            for (int trial = 1; trial <= MAX_TRIES; trial++)
            {
                run_simulation(grid_size, iterations, my_rank, number_of_processes, &reader);

                if (is_global_center(grid_size, number_of_processes, my_rank))
                {
                    fprintf(file, "%d, %d, %d, %d, %lf, %lf, %lf, %lf\n", trial, number_of_processes, grid_size, iterations, reader[0], reader[1], reader[2], reader[3]);
                    fflush(file);
                }
            }

    fclose(file);

    MPI_Finalize();
    return 0;
}
