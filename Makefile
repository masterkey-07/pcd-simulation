# COMPILATION
CC = gcc
CFLAGS = -Wall -Wextra -g -fopenmp -I. -Iinclude -O3

# FOLDERS
SRC_DIR = src
DIST_DIR = dist

# SRC
SAMPLE = sample
SEQUENCIAL = sequencial
PARALLEL_OMP = parallel_omp

# TARGET
SAMPLE_TARGET = $(DIST_DIR)/$(SAMPLE)
SEQUENCIAL_TARGET = $(DIST_DIR)/$(SEQUENCIAL)
PARALLEL_OMP_TARGET = $(DIST_DIR)/$(PARALLEL_OMP)

all: $(DIST_DIR) $(SAMPLE_TARGET) $(SEQUENCIAL_TARGET) $(PARALLEL_OMP_TARGET)

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

$(PARALLEL_OMP_TARGET): $(SRC_DIR)/$(PARALLEL_OMP).c | $(DIST_DIR)
	@echo "Compiling $(SRC_DIR)/$(PARALLEL_OMP).c into $(PARALLEL_OMP_TARGET)..."
	@$(CC) $(CFLAGS) $< -o $@
	@echo "Build complete: $(PARALLEL_OMP_TARGET)"

sequencial: $(SEQUENCIAL_TARGET)
	@echo "Running $(SEQUENCIAL_TARGET)..."
	@time ./$(SEQUENCIAL_TARGET)

omp: $(PARALLEL_OMP_TARGET)
	@echo "Running $(PARALLEL_OMP_TARGET)..."
	@time ./$(PARALLEL_OMP_TARGET)

sample: $(SAMPLE_TARGET)
	@echo "Running $(SAMPLE_TARGET)..."
	@time ./$(SAMPLE_TARGET)

clean:
	@echo "Cleaning up build artifacts..."
	@rm -rf $(DIST_DIR)
	@echo "Cleanup complete."

.PHONY: all clean sample sequencial omp run
