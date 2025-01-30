#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define START_ITERATIONS 500
#define MAX_ITERATIONS 1500
#define ITERATIONS_STEP 500

#define START_GRID_SIZE 2000
#define MAX_GRID_SIZE 6000
#define GRID_SIZE_STEP 2000

#define START_BLOCK_SIZE 4
#define MAX_BLOCK_SIZE 32
#define BLOCK_SIZE_STEP 2

#define MAX_TRIES 3

#define DELTA_T 0.01
#define DELTA_X 1.0
#define DIFFUSION_COEFFICIENT 0.1

__global__ void run_diff_equation_on_grid(float *input_grid, float *output_grid, int grid_size, int block_size)
{
    extern __shared__ float shared[];

    int shared_size = block_size + 2;

    int grid_x = blockIdx.x * block_size + threadIdx.x;
    int grid_y = blockIdx.y * block_size + threadIdx.y;

    int x = threadIdx.x + 1;
    int y = threadIdx.y + 1;

    if (grid_x < grid_size && grid_y < grid_size)
        shared[shared_size * y + x] = input_grid[grid_y * grid_size + grid_x];
    else
        shared[shared_size * y + x] = 0.0f;

    if (threadIdx.x == 0)
    {
        shared[shared_size * y] = (grid_x > 0) ? input_grid[grid_y * grid_size + (grid_x - 1)] : 0.0f;
        shared[shared_size * (y + 1) - 1] = (grid_x + block_size < grid_size) ? input_grid[grid_y * grid_size + (grid_x + block_size)] : 0.0f;
    }

    if (threadIdx.y == 0)
    {
        shared[x] = (grid_y > 0) ? input_grid[(grid_y - 1) * grid_size + grid_x] : 0.0f;
        shared[shared_size * (shared_size - 1) + x] = (grid_y + block_size < grid_size) ? input_grid[(grid_y + block_size) * grid_size + grid_x] : 0.0f;
    }

    __syncthreads();

    if (grid_x > 0 && grid_x < grid_size - 1 && grid_y > 0 && grid_y < grid_size - 1)
        output_grid[grid_size * grid_y + grid_x] = shared[shared_size * y + x] + DIFFUSION_COEFFICIENT * DELTA_T * ((shared[shared_size * (y + 1) + x] + shared[shared_size * (y - 1) + x] + shared[shared_size * y + (x + 1)] + shared[shared_size * y + (x - 1)] - 4 * shared[shared_size * y + x]) / (DELTA_X * DELTA_X));
}

int run_simulation(int size, int iterations, int block_size, float (*reader)[4])
{
    size_t total_size = size * size;

    cudaEvent_t runstart, runend;
    clock_t setupstart, setupend, teardownstart, teardownend;

    int grid_memory_size = total_size * sizeof(float);
    int number_of_blocks = (size + block_size - 1) / block_size;
    int shared_memory_size = (block_size + 2) * (block_size + 2) * sizeof(float);
    int grid_middle_index = total_size % 2 == 0 ? (total_size / 2) - (size / 2) : total_size / 2;

    float runtime_in_microseconds;
    float *device_input_grid, *device_output_grid;
    float *host_input_grid = (float *)malloc(grid_memory_size);
    float *host_output_grid = (float *)malloc(grid_memory_size);

    setupstart = clock();

    for (int i = 0; i < total_size; i++)
        host_input_grid[i] = 0.;

    host_input_grid[grid_middle_index] = 1.0;

    cudaMalloc(&device_input_grid, grid_memory_size);
    cudaMalloc(&device_output_grid, grid_memory_size);

    cudaMemcpy(device_input_grid, host_input_grid, grid_memory_size, cudaMemcpyHostToDevice);

    dim3 device_block_size(block_size, block_size);
    dim3 device_grid_size(number_of_blocks, number_of_blocks);

    cudaEventCreate(&runstart);
    cudaEventCreate(&runend);

    cudaEventRecord(runstart);

    setupend = clock();

    for (int i = 0; i < iterations; i++)
    {
        if (i % 2 == 0)
            run_diff_equation_on_grid<<<device_grid_size, device_block_size, shared_memory_size>>>(device_input_grid, device_output_grid, size, block_size);
        else
            run_diff_equation_on_grid<<<device_grid_size, device_block_size, shared_memory_size>>>(device_output_grid, device_input_grid, size, block_size);

        cudaDeviceSynchronize();
    }

    cudaMemcpy(host_output_grid, device_input_grid, grid_memory_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(runend);
    cudaEventSynchronize(runend);

    cudaEventElapsedTime(&runtime_in_microseconds, runstart, runend);

    teardownstart = clock();

    (*reader)[0] = host_output_grid[grid_middle_index];

    free(host_input_grid);
    free(host_output_grid);
    cudaFree(device_input_grid);
    cudaFree(device_output_grid);
    cudaEventDestroy(runstart);
    cudaEventDestroy(runend);

    teardownend = clock();

    (*reader)[1] = (float)(setupend - setupstart) / CLOCKS_PER_SEC;
    (*reader)[2] = runtime_in_microseconds / 1000;
    (*reader)[3] = (float)(teardownend - teardownstart) / CLOCKS_PER_SEC;

    return 0;
}

int main()
{

    int result;
    float reader[4];

    FILE *output = fopen("parallel-cuda.csv", "w");

    fprintf(output, "try,block_size,grid_size,iterations,concentration,setuptime,runtime,teardowntime\n");
    printf("try,block_size,grid_size,iterations,concentration,setuptime,runtime,teardowntime\n");

    for (int size = START_GRID_SIZE; size <= MAX_GRID_SIZE; size = size + GRID_SIZE_STEP)
        for (int iteration = START_ITERATIONS; iteration <= MAX_ITERATIONS; iteration = iteration + ITERATIONS_STEP)
            for (int block_size = START_BLOCK_SIZE; block_size <= MAX_BLOCK_SIZE; block_size = block_size * BLOCK_SIZE_STEP)
                for (int tryout = 0; tryout < MAX_TRIES; tryout++)
                {
                    result = run_simulation(size, iteration, block_size, &reader);

                    if (result == 1)
                        return 1;

                    printf("%d,%d,%d,%d,%f,%f,%f,%f\n", tryout, block_size, size, iteration, reader[0], reader[1], reader[2], reader[3]);
                    fprintf(output, "%d,%d,%d,%d,%f,%f,%f,%f\n", tryout, block_size, size, iteration, reader[0], reader[1], reader[2], reader[3]);

                    fflush(output);
                }

    fclose(output);

    return 0;
}