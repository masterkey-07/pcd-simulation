#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

double diffusion_equation(double **grid, int row, int column)
{
    return grid[row][column] + DIFFUSION_COEFFICIENT * DELTA_T * ((grid[row + 1][column] + grid[row - 1][column] + grid[row][column + 1] + grid[row][column - 1] - 4 * grid[row][column]) / (DELTA_X * DELTA_X));
}

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

    for (int row = 0; row < size; row++)
        for (int column = 0; column < size; column++)
            grid[row][column] = 0.;

    return grid;
}

void free_grid(double **grid, int size)
{
    for (int index = 0; index < size; index++)
        free(grid[index]);

    free(grid);
}

void run_diffusion_equation_on_grid(double **start_grid, double **next_grid, int grid_size, int iterations)
{
    double **helper;

    for (int t = 0; t < iterations; t++)
    {
        for (int i = 1; i < grid_size - 1; i++)
            for (int j = 1; j < grid_size - 1; j++)
                next_grid[i][j] = diffusion_equation(start_grid, i, j);

        helper = start_grid;
        start_grid = next_grid;
        next_grid = helper;
    }
}

int run_simulation(int size, int iterations, double (*reader)[4])
{
    clock_t setupstart, setupend, runstart, runend, teardownstart, teardownend;

    double **start_grid, **next_grid;

    setupstart = clock();

    start_grid = create_grid(size);

    if (start_grid == NULL)
        return 1;

    next_grid = create_grid(size);

    if (next_grid == NULL)
        return 1;

    start_grid[size / 2][size / 2] = 1.0;

    setupend = clock();

    (*reader)[1] = (double)(setupend - setupstart) / CLOCKS_PER_SEC;

    runstart = clock();

    run_diffusion_equation_on_grid(start_grid, next_grid, size, iterations);

    (*reader)[0] = start_grid[size / 2][size / 2];

    runend = clock();

    (*reader)[2] = (double)(runend - runstart) / CLOCKS_PER_SEC;

    teardownstart = clock();

    free_grid(start_grid, size);
    free_grid(next_grid, size);

    teardownend = clock();

    (*reader)[3] = (double)(teardownend - teardownstart) / CLOCKS_PER_SEC;

    return 0;
}

int main()
{
    double result;
    double reader[4];

    FILE *output = fopen("sequencial.csv", "w");

    fprintf(output, "try,grid_size,iterations,concentration,setuptime,runtime,teardowntime\n");

    for (int size = START_GRID_SIZE; size <= MAX_GRID_SIZE; size = size + GRID_SIZE_STEP)
        for (int iteration = START_ITERATIONS; iteration <= MAX_ITERATIONS; iteration = iteration + ITERATIONS_STEP)
            for (int try = 0; try < MAX_TRIES; try++)
            {
                result = run_simulation(size, iteration, &reader);

                if (result == 1)
                    return 1;

                fprintf(output, "%d,%d,%d,%lf,%lf,%lf,%lf\n", try, size, iteration, reader[0], reader[1], reader[2], reader[3]);

                fflush(output);
            }

    fclose(output);

    return 0;
}