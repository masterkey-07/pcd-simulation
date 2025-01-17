#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define RADIUS 1
#define BLOCK_SIZE 8

#define DELTA_T 0.01
#define DELTA_X 1.0
#define DIFFUSION_COEFFICIENT 0.1

// Kernel CUDA para aplicar stencil 2D
__global__ void
run_diff_equation_on_grid(const double *input_grid, double *output_grid, int size)
{
    __shared__ double shared_grid[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // Coordenadas globais
    int globalX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int globalY = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Coordenadas locais
    int localX = threadIdx.x + 1;
    int localY = threadIdx.y + 1;

    // Carregar os dados principais para a memória compartilhada
    if (globalX < size && globalY < size)
        shared_grid[localY][localX] = input_grid[globalY * size + globalX];
    else
        shared_grid[localY][localX] = 0.0f;

    // Carregar as bordas
    if (threadIdx.x == 0)
    {
        shared_grid[localY][threadIdx.x] = (globalX > 0) ? input_grid[globalY * size + (globalX - 1)] : 0.0f;
        shared_grid[localY][threadIdx.x + BLOCK_SIZE + 1] = (globalX + BLOCK_SIZE < size) ? input_grid[globalY * size + (globalX + BLOCK_SIZE)] : 0.0f;
    }

    if (threadIdx.y == 0)
    {
        shared_grid[threadIdx.y][localX] = (globalY > 0) ? input_grid[(globalY - 1) * size + globalX] : 0.0f;
        shared_grid[threadIdx.y + BLOCK_SIZE + 1][localX] = (globalY + BLOCK_SIZE < size) ? input_grid[(globalY + BLOCK_SIZE) * size + globalX] : 0.0f;
    }

    __syncthreads();

    // Aplicar stencil
    if (globalX > 0 && globalX < size - 1 && globalY > 0 && globalY < size - 1)
        output_grid[globalY * size + globalX] = shared_grid[localY][localX] + DIFFUSION_COEFFICIENT * DELTA_T * ((shared_grid[localY + 1][localX] + shared_grid[localY - 1][localX] + shared_grid[localY][localX + 1] + shared_grid[localY][localX - 1] - 4 * shared_grid[localY][localX]) / (DELTA_X * DELTA_X));
}

void printMatrix(const char *label, double *matrix, int size)
{
    printf("%s:\n", label);
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%lf ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

int main()
{

    int iterations = 1000;
    size_t size = 2000;
    size_t total_size = size * size;

    // Alocar e inicializar a matriz no host
    double *input_grid = (double *)malloc(total_size * sizeof(double));
    double *output_grid = (double *)malloc(total_size * sizeof(double));

    for (int i = 0; i < total_size; i++)
        input_grid[i] = 0.;

    input_grid[total_size % 2 == 0 ? (total_size / 2) - (size / 2) : total_size / 2] = 1.0;

    // Alocar memória no dispositivo
    double *cuda_input_grid, *cuda_output_grid;
    cudaMalloc(&cuda_input_grid, total_size * sizeof(double));
    cudaMalloc(&cuda_output_grid, total_size * sizeof(double));

    // Copiar a matriz de entrada para o dispositivo
    cudaMemcpy(cuda_input_grid, input_grid, total_size * sizeof(double), cudaMemcpyHostToDevice);

    int cuda_grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Configurar dimensões de bloco e grid
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(cuda_grid_size, cuda_grid_size);

    for (int i = 0; i < iterations; i++)
        if (i % 2 == 0)
            run_diff_equation_on_grid<<<gridSize, blockSize>>>(cuda_input_grid, cuda_output_grid, size);
        else
            run_diff_equation_on_grid<<<gridSize, blockSize>>>(cuda_output_grid, cuda_input_grid, size);

    // Copiar o resultado de volta para o host
    cudaMemcpy(output_grid, cuda_input_grid, total_size * sizeof(double), cudaMemcpyDeviceToHost);

    printf("%lf", output_grid[total_size % 2 == 0 ? (total_size / 2) - (size / 2) : total_size / 2]);

    // Liberar memória
    free(input_grid);
    free(output_grid);
    cudaFree(cuda_input_grid);
    cudaFree(cuda_output_grid);

    return 0;
}