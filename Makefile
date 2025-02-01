# COMPILATION
CC = gcc
MPICC = mpicc
CFLAGS = -Wall -Wextra -g -fopenmp -I. -Iinclude -O3
CFLAGS2 = -Wall -Wextra -g -fopenmp -I. -Iinclude

# FOLDERS
SRC_DIR = src
DIST_DIR = dist

# SRC
SAMPLE = sample
SEQUENCIAL = sequencial
PARALLEL_OMP = parallel_omp
PARALLEL_MPI = parallel_mpi

# TARGET

SAMPLE_TARGET = $(DIST_DIR)/$(SAMPLE)
SEQUENCIAL_TARGET = $(DIST_DIR)/$(SEQUENCIAL)
PARALLEL_OMP_TARGET = $(DIST_DIR)/$(PARALLEL_OMP)
PARALLEL_MPI_TARGET = $(DIST_DIR)/$(PARALLEL_MPI)

all: $(DIST_DIR) $(SAMPLE_TARGET) $(SEQUENCIAL_TARGET) $(SEQUENCIAL_TARGET)_raw $(PARALLEL_OMP_TARGET) $(PARALLEL_MPI_TARGET)

$(DIST_DIR):
	@mkdir -p $(DIST_DIR)
	@echo "Created directory: $(DIST_DIR)"

$(SAMPLE_TARGET): $(SRC_DIR)/$(SAMPLE).c | $(DIST_DIR)
	@echo "Compiling $(SRC_DIR)/$(SAMPLE).c into $(SAMPLE_TARGET)..."
	@$(CC) $(CFLAGS) $< -o $@
	@echo "Build complete: $(SAMPLE_TARGET)"

$(SEQUENCIAL_TARGET): $(SRC_DIR)/$(SEQUENCIAL).c | $(DIST_DIR)
	@echo "Compiling $(SRC_DIR)/$(SEQUENCIAL).c into $(SEQUENCIAL_TARGET)..."
	@$(CC) $(CFLAGS) $< -o $@
	@echo "Build complete: $(SEQUENCIAL_TARGET)"

$(SEQUENCIAL_TARGET)_raw: $(SRC_DIR)/$(SEQUENCIAL).c | $(DIST_DIR)
	@echo "Compiling $(SRC_DIR)/$(SEQUENCIAL).c into $(SEQUENCIAL_TARGET)_raw..."
	@$(CC) $(CFLAGS2) $< -o $@
	@echo "Build complete: $(SEQUENCIAL_TARGET)_raw"

$(PARALLEL_OMP_TARGET): $(SRC_DIR)/$(PARALLEL_OMP).c | $(DIST_DIR)
	@echo "Compiling $(SRC_DIR)/$(PARALLEL_OMP).c into $(PARALLEL_OMP_TARGET)..."
	@$(CC) $(CFLAGS2) $< -o $@
	@echo "Build complete: $(PARALLEL_OMP_TARGET)"

$(PARALLEL_MPI_TARGET): $(SRC_DIR)/$(PARALLEL_MPI).c | $(DIST_DIR)
	@echo "Compiling $(SRC_DIR)/$(PARALLEL_MPI).c into $(PARALLEL_MPI_TARGET)..."
	@$(MPICC) $< -o $@ -ldl -lm
	@echo "Build complete: $(PARALLEL_MPI_TARGET)"

sequencial: $(SEQUENCIAL_TARGET)
	@echo "Running $(SEQUENCIAL_TARGET)..."
	@time ./$(SEQUENCIAL_TARGET)

omp: $(PARALLEL_OMP_TARGET)
	@echo "Running $(PARALLEL_OMP_TARGET)..."
	@time ./$(PARALLEL_OMP_TARGET)

mpi: $(PARALLEL_MPI_TARGET)
	@echo "Running $(PARALLEL_MPI_TARGET)..."
	@time mpirun --allow-run-as-root --oversubscribe -np 12 ./$(PARALLEL_MPI_TARGET)

sample: $(SAMPLE_TARGET)
	@echo "Running $(SAMPLE_TARGET)..."
	@time ./$(SAMPLE_TARGET)

clean:
	@echo "Cleaning up build artifacts..."
	@rm -rf $(DIST_DIR)
	@echo "Cleanup complete."

.PHONY: all clean sample sequencial mpi omp run