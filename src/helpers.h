#include <math.h>
#include "tensor.h"
#include <stdbool.h>

#define EPSILON 1e-6
#define MAX_PRINT_SIZE 10

// tensor helpers
int get_size(int shape[], int ndim);
void set_cst_data(float* data, int size, float value);
void set_data(float* data_src, int size_src, float* data_dst, int size_dst);
void print_data(float* data, int shape[], int ndim);
bool is_equal_data(float* data_a, float* data_b, int size);
bool is_same_shape(int shape_a[], int shape_b[], int ndim_a, int ndim_b);
