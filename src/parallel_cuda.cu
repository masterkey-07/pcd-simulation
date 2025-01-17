#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define RADIUS 1

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

int main()
{

    int iterations = 1000;
    size_t size = 2000;
    size_t total_size = size * size;

    int block_size = 8;

    int grid_memory_size = total_size * sizeof(float);
    int shared_memory_size = (block_size + 2) * (block_size + 2) * sizeof(float);

    float *host_input_grid = (float *)malloc(grid_memory_size);
    float *host_output_grid = (float *)malloc(grid_memory_size);

    for (int i = 0; i < total_size; i++)
        host_input_grid[i] = 0.;

    host_input_grid[total_size % 2 == 0 ? (total_size / 2) - (size / 2) : total_size / 2] = 1.0;

    float *device_input_grid, *device_output_grid;
    cudaMalloc(&device_input_grid, grid_memory_size);
    cudaMalloc(&device_output_grid, grid_memory_size);

    cudaMemcpy(device_input_grid, host_input_grid, grid_memory_size, cudaMemcpyHostToDevice);

    int number_of_blocks = (size + block_size - 1) / block_size;

    dim3 device_block_size(block_size, block_size);
    dim3 device_grid_size(number_of_blocks, number_of_blocks);

    for (int i = 0; i < iterations; i++)
        if (i % 2 == 0)
            run_diff_equation_on_grid<<<device_grid_size, device_block_size, shared_memory_size>>>(device_input_grid, device_output_grid, size, block_size);
        else
            run_diff_equation_on_grid<<<device_grid_size, device_block_size, shared_memory_size>>>(device_output_grid, device_input_grid, size, block_size);

    cudaMemcpy(host_output_grid, device_input_grid, grid_memory_size, cudaMemcpyDeviceToHost);

    printf("%f", host_output_grid[total_size % 2 == 0 ? (total_size / 2) - (size / 2) : total_size / 2]);

    free(host_input_grid);
    free(host_output_grid);
    cudaFree(device_input_grid);
    cudaFree(device_output_grid);

    return 0;
}