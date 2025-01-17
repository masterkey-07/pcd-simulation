#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define RADIUS 1
#define BLOCK_SIZE 8

// Kernel CUDA para aplicar stencil 2D
__global__ void
stencil2D_shared(const float *input, float *output, int width, int height)
{
    __shared__ float sharedMem[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];

    // Coordenadas globais
    int globalX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int globalY = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Coordenadas locais
    int localX = threadIdx.x + RADIUS;
    int localY = threadIdx.y + RADIUS;

    // Carregar os dados principais para a memória compartilhada
    if (globalX < width && globalY < height)
        sharedMem[localY][localX] = input[globalY * width + globalX];
    else
        sharedMem[localY][localX] = 0.0f;

    // Carregar as bordas
    if (threadIdx.x < RADIUS)
    {
        sharedMem[localY][threadIdx.x] = (globalX >= RADIUS) ? input[globalY * width + (globalX - RADIUS)] : 0.0f;
        sharedMem[localY][threadIdx.x + BLOCK_SIZE + RADIUS] = (globalX + BLOCK_SIZE < width) ? input[globalY * width + (globalX + BLOCK_SIZE)] : 0.0f;
    }

    if (threadIdx.y < RADIUS)
    {
        sharedMem[threadIdx.y][localX] = (globalY >= RADIUS) ? input[(globalY - RADIUS) * width + globalX] : 0.0f;
        sharedMem[threadIdx.y + BLOCK_SIZE + RADIUS][localX] = (globalY + BLOCK_SIZE < height) ? input[(globalY + BLOCK_SIZE) * width + globalX] : 0.0f;
    }

    __syncthreads();

    // Aplicar stencil
    if (globalX < width && globalY < height)
    {
        float sum = 0.0f;
        for (int dy = -RADIUS; dy <= RADIUS; dy++)
        {
            for (int dx = -RADIUS; dx <= RADIUS; dx++)
            {
                sum += sharedMem[localY + dy][localX + dx];
            }
        }
        output[globalY * width + globalX] = sum / ((2 * RADIUS + 1) * (2 * RADIUS + 1));
    }
}

void printMatrix(const char *label, float *matrix, int width, int height)
{
    printf("%s:\n", label);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%6.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main()
{
    // Dimensões da matriz
    int width = 16;
    int height = 16;

    // Tamanho em bytes
    size_t size = width * height * sizeof(float);

    // Alocar e inicializar a matriz no host
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    for (int i = 0; i < width * height; i++)
        h_input[i] = rand() % 10 + 1; // Valores aleatórios entre 1 e 10

    // Alocar memória no dispositivo
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copiar a matriz de entrada para o dispositivo
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int grid_width = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_height = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Configurar dimensões de bloco e grid
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(grid_width, grid_height);

    // Chamar o kernel
    stencil2D_shared<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // Copiar o resultado de volta para o host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Imprimir matrizes de entrada e saída
    printMatrix("Matriz de entrada", h_input, width, height);
    printMatrix("Matriz de saída (após stencil)", h_output, width, height);

    // Liberar memória
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
