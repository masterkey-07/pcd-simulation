#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define GRID_SIZE 2000
#define ITERATIONS 500
#define DIFFUSION_COEFFICIENT 0.1
#define DELTA_T 0.01
#define DELTA_X 1.0

double diffusion_equation(double **grid, int row, int column)
{
    return grid[row][column] + DIFFUSION_COEFFICIENT * DELTA_T * ((grid[row + 1][column] + grid[row - 1][column] + grid[row][column + 1] + grid[row][column - 1] - 4 * grid[row][column]) / (DELTA_X * DELTA_X));
}

void run_diffusion_equation_on_grid(double **initial_concetration_grid, double **next_concentration_grid)
{

    omp_set_num_threads(12);

    for (int iteration = 0; iteration < ITERATIONS; iteration++)
    {

#pragma omp parallel for collapse(2)
        for (int row = 1; row < GRID_SIZE - 1; row++)
            for (int column = 1; column < GRID_SIZE - 1; column++)
                next_concentration_grid[row][column] = diffusion_equation(initial_concetration_grid, row, column);

        double average_diffusion = 0., result;

#pragma omp parallel for collapse(2)
        for (int i = 1; i < GRID_SIZE - 1; i++)
            for (int j = 1; j < GRID_SIZE - 1; j++)
            {
                result = fabs(next_concentration_grid[i][j] - initial_concetration_grid[i][j]);
#pragma omp critical
                {
                    average_diffusion += result;
                }

                initial_concetration_grid[i][j] = next_concentration_grid[i][j];
            }

        if ((iteration % 100) == 0)
            printf("Iteração %d: Diferença = %g\n", iteration, average_diffusion / ((GRID_SIZE - 2) * (GRID_SIZE - 2)));
    }
}

double **create_concentration_grid()
{
    double **grid = (double **)malloc(GRID_SIZE * sizeof(double *));

    if (grid == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    for (int index = 0; index < GRID_SIZE; index++)
    {
        grid[index] = (double *)malloc(GRID_SIZE * sizeof(double));

        if (grid[index] == NULL)
        {
            fprintf(stderr, "Memory allocation failed\n");
            return NULL;
        }
    }

#pragma omp parallel for collapse(2)
    for (int row = 0; row < GRID_SIZE; row++)
        for (int column = 0; column < GRID_SIZE; column++)
            grid[row][column] = 0.;

    return grid;
}

int main()
{
    double start, end;

    double **initial_concentration_grid = create_concentration_grid();

    if (initial_concentration_grid == NULL)
        return 1;

    double **next_concentration_grid = create_concentration_grid();

    if (next_concentration_grid == NULL)
        return 1;

    initial_concentration_grid[GRID_SIZE / 2][GRID_SIZE / 2] = 1.0;

    start = omp_get_wtime();

    run_diffusion_equation_on_grid(initial_concentration_grid, next_concentration_grid);

    end = omp_get_wtime();

    printf("(%lfs) Concentração final no centro: %f\n", end - start, initial_concentration_grid[GRID_SIZE / 2][GRID_SIZE / 2]);

    return 0;
}