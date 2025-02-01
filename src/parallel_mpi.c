#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Simulation parameters.
#define ITERATIONS 100

// Global simulation constants.
const int GLOBAL_N = 101;   // Global matrix size (square matrix). Use an odd number so there is a unique center.
const float D = 0.1f;       // Diffusion constant.
const float DELTA_T = 0.01f;// Time step.
const float DELTA_X = 1.0f; // Spatial resolution.

// Macro to index into a 1D array that represents a 2D array.
#define IDX(i, j, ncols) ((i) * (ncols) + (j))

// A helper to decide if a given global index is a boundary cell.
bool is_global_boundary(int i, int j) {
    return (i == 0 || i == GLOBAL_N - 1 || j == 0 || j == GLOBAL_N - 1);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Partition the rows among processes ---
    // We use a block decomposition that may result in uneven partitions.
    int base_rows = GLOBAL_N / size;
    int remainder = GLOBAL_N % size;
    int local_rows = base_rows + (rank < remainder ? 1 : 0);

    // Determine the global starting row for this process.
    int row_start = rank * base_rows + (rank < remainder ? rank : remainder);
    // Now, row_start is the index of the first global row for this process.

    // --- Allocate arrays ---
    // We allocate (local_rows+2) rows to include two ghost rows (top and bottom).
    int total_rows = local_rows + 2;
    float *current = (float *)malloc(total_rows * GLOBAL_N * sizeof(float));
    float *next    = (float *)malloc(total_rows * GLOBAL_N * sizeof(float));
    if (!current || !next) {
        fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- Initialize the local matrix ---
    // Set all interior cells to 0.
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 0; j < GLOBAL_N; j++) {
            current[IDX(i, j, GLOBAL_N)] = 0.0f;
        }
    }
    // Initialize ghost rows (top and bottom) to 0.
    for (int j = 0; j < GLOBAL_N; j++) {
        current[IDX(0, j, GLOBAL_N)] = 0.0f;             // Top ghost row.
        current[IDX(local_rows + 1, j, GLOBAL_N)] = 0.0f;  // Bottom ghost row.
    }
    // Set the center cell to 1 if it is contained in this process.
    int center = GLOBAL_N / 2; // Global center index (both row and column).
    if (center >= row_start && center < row_start + local_rows) {
        // The corresponding local row index (within the interior part) is:
        int local_center = center - row_start + 1;  // +1 accounts for the ghost row at index 0.
        current[IDX(local_center, center, GLOBAL_N)] = 1.0f;
    }

    // Copy current into next (initial condition).
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < GLOBAL_N; j++) {
            next[IDX(i, j, GLOBAL_N)] = current[IDX(i, j, GLOBAL_N)];
        }
    }

    // --- Main iterative loop ---
    for (int iter = 0; iter < ITERATIONS; iter++) {
        MPI_Request reqs[4];
        int req_count = 0;

        // --- Exchange halo rows with neighbors ---
        // Upward exchange: if there is a process above, send row 1 and receive into ghost row 0.
        if (rank > 0) {
            MPI_Isend(&current[IDX(1, 0, GLOBAL_N)], GLOBAL_N, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(&current[IDX(0, 0, GLOBAL_N)], GLOBAL_N, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        // Downward exchange: if there is a process below, send last real row (row local_rows) and receive into ghost row local_rows+1.
        if (rank < size - 1) {
            MPI_Isend(&current[IDX(local_rows, 0, GLOBAL_N)], GLOBAL_N, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(&current[IDX(local_rows + 1, 0, GLOBAL_N)], GLOBAL_N, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        if (req_count > 0)
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

        // --- Update each interior cell ---
        // Loop over the real rows in the local block (indices 1 to local_rows).
        for (int i = 1; i <= local_rows; i++) {
            int global_i = row_start + (i - 1);  // Convert local interior row to global row index.
            for (int j = 0; j < GLOBAL_N; j++) {
                // For global boundary cells, we leave the value unchanged.
                if (is_global_boundary(global_i, j)) {
                    next[IDX(i, j, GLOBAL_N)] = current[IDX(i, j, GLOBAL_N)];
                } else {
                    // Compute the laplacian (using the four cardinal neighbors).
                    float laplacian = (current[IDX(i + 1, j, GLOBAL_N)] +
                                       current[IDX(i - 1, j, GLOBAL_N)] +
                                       current[IDX(i, j + 1, GLOBAL_N)] +
                                       current[IDX(i, j - 1, GLOBAL_N)] -
                                       4.0f * current[IDX(i, j, GLOBAL_N)]) / (DELTA_X * DELTA_X);
                    // Update the cell using the given formula.
                    next[IDX(i, j, GLOBAL_N)] = current[IDX(i, j, GLOBAL_N)] + D * DELTA_T * laplacian;
                }
            }
        }

        // Swap the buffers for the next iteration.
        float *tmp = current;
        current = next;
        next = tmp;
    }

    // --- Compute the global mean of all cells ---
    double local_sum = 0.0;
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 0; j < GLOBAL_N; j++) {
            local_sum += current[IDX(i, j, GLOBAL_N)];
        }
    }
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Process 0 computes and prints the final mean.
    if (rank == 0) {
        double mean = global_sum / (GLOBAL_N * GLOBAL_N);
        printf("Final mean value after %d iterations: %f\n", ITERATIONS, mean);
    }

    free(current);
    free(next);
    MPI_Finalize();
    return 0;
}
