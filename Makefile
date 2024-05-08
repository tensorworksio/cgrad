CC = gcc
CFLAGS = -Isrc -Wall -Wextra -g
LDFLAGS = -lm -lcriterion -lcsptr
SRC = $(wildcard main.c)
OBJ = $(SRC:.c=.o)
TEST_SRC = $(wildcard tests/*.c)
TEST_OBJ = $(TEST_SRC:.c=.o)
TEST_BIN = $(TEST_SRC:.c=)

.PHONY: all clean test main

all: main

main: main.o $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

main.o: main.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST_BIN)
	for test in $(TEST_BIN); do echo "Running $$test"; ./$$test; done

tests/%: tests/%.o $(filter-out main.o, $(OBJ))
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJ) $(TEST_OBJ) $(TEST_BIN) main.o main