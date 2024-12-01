#include <stdio.h>
#include <stdlib.h>

double **create_grid(int size)
{
    double **grid = (double **)malloc(size * sizeof(double *));

    if (grid == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    for (int index = 0; index < size; index++)
    {
        grid[index] = (double *)malloc(size * sizeof(double));

        if (grid[index] == NULL)
        {
            fprintf(stderr, "Memory allocation failed\n");
            return NULL;
        }
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