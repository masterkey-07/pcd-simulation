# COMPILATION
CC = gcc
CFLAGS = -Wall -Wextra -g -fopenmp -I. -Iinclude

# FOLDERS
SRC_DIR = src
DIST_DIR = dist
DATA_DIR = data

# SOURCE
SAMPLE = sample
MAIN = main	grid diffusion_equation
SRC = $(wildcard $(SRC_DIR)/*.c) 
OBJ = $(SRC:$(SRC_DIR)/%.c=$(DIST_DIR)/%.o)

MAIN_OBJ = $(MAIN:%=$(DIST_DIR)/%.o)


# TARGET
MAIN_TARGET = $(DIST_DIR)/main
SAMPLE_TARGET = $(DIST_DIR)/$(SAMPLE)

all: $(DIST_DIR) $(DATA_DIR) $(SAMPLE_TARGET) $(MAIN_TARGET)

$(DIST_DIR):
	@mkdir -p $(DIST_DIR)
	@echo "Created directory: $(DIST_DIR)"

$(DATA_DIR):
	@mkdir -p $(DATA_DIR)
	@echo "Created directory: $(DATA_DIR)"

$(SAMPLE_TARGET): $(SRC_DIR)/$(SAMPLE).c | $(DIST_DIR)
	@echo "Compiling $(SRC_DIR)/$(SAMPLE).c into $(SAMPLE_TARGET)..."
	@$(CC) $(CFLAGS) $< -o $@
	@echo "Build complete: $(SAMPLE_TARGET)"

$(MAIN_TARGET): $(MAIN_OBJ)
	@echo "Compiling $(MAIN_OBJ) into $(MAIN_TARGET)..."
	@$(CC) $(CFLAGS) -o $(MAIN_TARGET) $(MAIN_OBJ)
	@echo "Build complete: $(MAIN_TARGET)"

$(DIST_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $< into $@..."
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo "Object compiled: $@"

run: $(MAIN_TARGET)
	@echo "Running $(MAIN_TARGET)..."
	@time ./$(MAIN_TARGET)

sample: $(SAMPLE_TARGET)
	@echo "Running $(SAMPLE_TARGET)..."
	@time ./$(SAMPLE_TARGET)

clean:
	@echo "Cleaning up build artifacts..."
	@rm -rf $(DIST_DIR)
	@rm -rf $(DATA_DIR)
	@echo "Cleanup complete."

.PHONY: all clean sample run
