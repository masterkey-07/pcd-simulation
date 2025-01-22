#include <omp.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define START_THREADS 12
// #define START_THREADS 4
#define MAX_THREADS 12
#define THREADS_STEP 4

#define START_ITERATIONS 1000
// #define START_ITERATIONS 200
#define MAX_ITERATIONS 1000
#define ITERATIONS_STEP 200

#define MAX_TRIES 3

#define START_GRID_SIZE 4000
// #define START_GRID_SIZE 400
#define MAX_GRID_SIZE 4000
#define GRID_SIZE_STEP 400

#define DELTA_T 0.01
#define DELTA_X 1.0
#define DIFFUSION_COEFFICIENT 0.1

double **create_grid(int size)
{
    double **grid = (double **)malloc(size * sizeof(double *));

    if (grid == NULL)
        return NULL;

    for (int index = 0; index < size; index++)
    {
        grid[index] = (double *)malloc(size * sizeof(double));

        if (grid[index] == NULL)
            return NULL;
    }

#pragma omp parallel for
    for (int row = 0; row < size; row++)
        for (int column = 0; column < size; column++)
            grid[row][column] = 0.;

    return grid;
}

void free_grid(double **grid, int size)
{
#pragma omp parallel for
    for (int index = 0; index < size; index++)
        free(grid[index]);

    free(grid);
}

void run_diffusion_equation_on_grid(double **start_grid, double **next_grid, int grid_size, int iterations)
{
    double **helper;

    for (int iteration = 0; iteration < iterations; iteration++)
    {

#pragma omp parallel for
        for (int row = 1; row < grid_size - 1; row++)
            for (int column = 1; column < grid_size - 1; column++)
                next_grid[row][column] = start_grid[row][column] + DIFFUSION_COEFFICIENT * DELTA_T * ((start_grid[row + 1][column] + start_grid[row - 1][column] + start_grid[row][column + 1] + start_grid[row][column - 1] - 4 * start_grid[row][column]) / (DELTA_X * DELTA_X));

        helper = start_grid;
        start_grid = next_grid;
        next_grid = helper;
    }
}

int run_simulation(int size, int iterations, int threads, double (*reader)[4])
{
    int maximum_threads = omp_get_num_procs();

    double setupstart, setupend, runstart, runend, teardownstart, teardownend, **start_grid, **next_grid;

    omp_set_num_threads(maximum_threads);

    setupstart = omp_get_wtime();

    start_grid = create_grid(size);

    if (start_grid == NULL)
        return 1;

    next_grid = create_grid(size);

    if (next_grid == NULL)
        return 1;

    start_grid[size / 2][size / 2] = 1.0;

    setupend = omp_get_wtime();

    runstart = omp_get_wtime();

    omp_set_num_threads(threads);

    run_diffusion_equation_on_grid(start_grid, next_grid, size, iterations);

    runend = omp_get_wtime();

    teardownstart = omp_get_wtime();

    (*reader)[0] = start_grid[size / 2][size / 2];

    free_grid(start_grid, size);
    free_grid(next_grid, size);

    teardownend = omp_get_wtime();

    (*reader)[1] = setupend - setupstart;
    (*reader)[2] = runend - runstart;
    (*reader)[3] = teardownend - teardownstart;

    return 0;
}

int main()
{
    int result;
    double reader[4];

    FILE *output = fopen("parallel-omp.csv", "w");

    fprintf(output, "try,thread,grid_size,iterations,concentration,setuptime,runtime,teardowntime\n");

    for (int threads = START_THREADS; threads <= MAX_THREADS; threads = threads + THREADS_STEP)
        for (int size = START_GRID_SIZE; size <= MAX_GRID_SIZE; size = size + GRID_SIZE_STEP)
            for (int iteration = START_ITERATIONS; iteration <= MAX_ITERATIONS; iteration = iteration + ITERATIONS_STEP)
                for (int try = 0; try < MAX_TRIES; try++)
                {
                    result = run_simulation(size, iteration, threads, &reader);

                    if (result == 1)
                        return 1;

                    fprintf(output, "%d,%d,%d,%d,%lf,%lf,%lf,%lf\n", try, threads, size, iteration, reader[0], reader[1], reader[2], reader[3]);

                    fflush(output);
                }

    fclose(output);

    return 0;
}