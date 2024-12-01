#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/grid.h"
#include "../include/diffusion_equation.h"

int run_simulation(int size, int iterations, int threads, double (*reader)[2])
{
    int maximum_threads = omp_get_num_procs();

    double start, end, **start_grid, **next_grid;

    omp_set_num_threads(maximum_threads);

    start_grid = create_grid(size);

    if (start_grid == NULL)
        return 1;

    next_grid = create_grid(size);

    if (next_grid == NULL)
        return 1;

    start_grid[size / 2][size / 2] = 1.0;

    start = omp_get_wtime();

    if (threads <= 1)
        run_sequencial_diffusion_equation_on_grid(start_grid, next_grid, size, iterations);
    else
    {
        omp_set_num_threads(threads);

        run_parallelized_diffusion_equation_on_grid(start_grid, next_grid, size, iterations);
    }

    end = omp_get_wtime();

    (*reader)[0] = end - start;
    (*reader)[1] = start_grid[size / 2][size / 2];

    omp_set_num_threads(maximum_threads);

    free_grid(start_grid, size);
    free_grid(next_grid, size);

    return 0;
}

int main()
{
    double reader[2];

    int result, maximum_threads = omp_get_num_procs();

    FILE *output = fopen("data/output.csv", "w");

    fprintf(output, "threads,grid_size,iterations,time,concentration\n");

    for (int thread = 2; thread <= maximum_threads; thread = thread + 1)
        for (int size = 2000; size <= 4000; size = size + 1000)
            for (int iteration = 300; iteration <= 1200; iteration = iteration + 300)
            {
                result = run_simulation(size, iteration, thread, &reader);

                if (result == 1)
                    return 1;

                fprintf(output, "%d,%d,%d,%lf,%lf\n", thread, size, iteration, reader[0], reader[1]);

                fflush(output);
            }

    fclose(output);

    return 0;
}