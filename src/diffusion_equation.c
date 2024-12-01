#include <omp.h>
#include <math.h>

#define DIFFUSION_COEFFICIENT 0.1
#define DELTA_T 0.01
#define DELTA_X 1.0

double diffusion_equation(double **grid, int row, int column)
{
    return grid[row][column] + DIFFUSION_COEFFICIENT * DELTA_T * ((grid[row + 1][column] + grid[row - 1][column] + grid[row][column + 1] + grid[row][column - 1] - 4 * grid[row][column]) / (DELTA_X * DELTA_X));
}

void run_sequencial_diffusion_equation_on_grid(double **start_grid, double **next_grid, int grid_size, int iterations)
{
    for (int t = 0; t < iterations; t++)
    {
        for (int i = 1; i < grid_size - 1; i++)
            for (int j = 1; j < grid_size - 1; j++)
                next_grid[i][j] = diffusion_equation(start_grid, i, j);

        double average_diffusion = 0.;

        for (int i = 1; i < grid_size - 1; i++)
            for (int j = 1; j < grid_size - 1; j++)
            {
                average_diffusion += fabs(next_grid[i][j] - start_grid[i][j]);
                start_grid[i][j] = next_grid[i][j];
            }
    }
}

void run_parallelized_diffusion_equation_on_grid(double **start_grid, double **next_grid, int grid_size, int iterations)
{
    for (int iteration = 0; iteration < iterations; iteration++)
    {

#pragma omp parallel for
        for (int row = 1; row < grid_size - 1; row++)
            for (int column = 1; column < grid_size - 1; column++)
                next_grid[row][column] = diffusion_equation(start_grid, row, column);

        double average_diffusion = 0.;

#pragma omp parallel for reduction(+ : average_diffusion)
        for (int i = 1; i < grid_size - 1; i++)
            for (int j = 1; j < grid_size - 1; j++)
            {
                average_diffusion += fabs(next_grid[i][j] - start_grid[i][j]);
                start_grid[i][j] = next_grid[i][j];
            }
    }
}
