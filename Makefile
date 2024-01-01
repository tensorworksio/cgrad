# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra -g

# Source files
SRCS = main.c tensor.c ops.c backops.c helpers.c logger.c

# Object files
OBJS = $(SRCS:.c=.o)

# Target executable
TARGET = main

# Default target
all: $(TARGET)

# Compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Link object files into the target executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ -lm

# Clean up object files and the target executable
clean:
	rm -f $(OBJS) $(TARGET)
