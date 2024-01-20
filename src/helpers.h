#include "tensor.h"
#include <math.h>
#include <stdbool.h>

#define EPSILON 1e-6
#define MAX_PRINT_SIZE 10

// tensor helpers
int get_size(int shape[], int ndim);
int get_index(int shape[], int coords[], int ndim);

void set_data(float *data, float value, int size);
void set_shape(int shape[], slice_t ranges[], int ndim);
void set_ranges(slice_t ranges[], int shape[], int ndim);

void print_metadata(int data[], int ndim);
void print_data(float *data, int shape[], int stride[], int ndim);

bool is_equal_data(float *data_a, float *data_b, int size);
bool is_same_shape(int shape_a[], int shape_b[], int ndim_a, int ndim_b);
