# COMPILATION
CC = gcc
CFLAGS = -Wall -Wextra -g -fopenmp

# FOLDERS
SRC_DIR = src
DIST_DIR = dist

# SOURCE
SAMPLE = sample
MAIN = main

# TARGET
MAIN_TARGET = $(DIST_DIR)/$(MAIN)
SAMPLE_TARGET = $(DIST_DIR)/$(SAMPLE)


all: $(DIST_DIR) $(MAIN_TARGET) $(SAMPLE_TARGET)

$(MAIN_TARGET): $(SRC_DIR)/$(MAIN).c | $(DIST_DIR)
	@echo "Compiling $(SRC_DIR)/$(MAIN).c into $(MAIN_TARGET)..."
	@$(CC) $(CFLAGS) $< -o $@
	@echo "Build complete: $(MAIN_TARGET)"

$(SAMPLE_TARGET): $(SRC_DIR)/$(SAMPLE).c | $(DIST_DIR)
	@echo "Compiling $(SRC_DIR)/$(SAMPLE).c into $(SAMPLE_TARGET)..."
	@$(CC) $(CFLAGS) $< -o $@
	@echo "Build complete: $(SAMPLE_TARGET)"

$(DIST_DIR):
	@mkdir -p $(DIST_DIR)
	@echo "Created directory: $(DIST_DIR)"

run: $(MAIN_TARGET)
	@echo "Running $(MAIN_TARGET)..."
	@time ./$(MAIN_TARGET)

sample: $(SAMPLE_TARGET)
	@echo "Running $(SAMPLE_TARGET)..."
	@time ./$(SAMPLE_TARGET)

clean:
	@echo "Cleaning up build artifacts..."
	@rm -rf $(DIST_DIR)
	@echo "Cleanup complete."

.PHONY: all clean sample run
