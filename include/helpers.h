#include "iterator.h"
#include "log.h"
#include "tensor.h"
#include <math.h>
#include <stdbool.h>

#define EPSILON 1e-6
#define MAX_PRINT_SIZE 10

// tensor helpers
size_t get_size (int shape[], size_t ndim);
int    get_index (int coords[], int shape[], size_t ndim);

void set_data (float *data, float value, size_t size);
void normalize_range (slice_t range[], int shape[], size_t ndim);

void print_metadata (int data[], size_t ndim);
void print_data (float *data, int shape[], int stride[], size_t ndim);

bool is_equal_data (float *data_a, float *data_b, size_t size);
bool is_same_shape (int shape_a[], int shape_b[], size_t ndim_a, size_t ndim_b);

void build_topo (tensor_t *root, tensor_t ***list, int *count, int *capacity);