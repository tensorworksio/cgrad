#include "tensor.h"
#include <stdbool.h>

#define MAX_PRINT_SIZE 10

// tensor helpers
int get_size(int shape[], int ndim);
bool same_shape(tensor_t* a, tensor_t* b);
void set_tensor_data(float* data, int size, float value);
void print_tensor_data(float* data, int shape[], int ndim);
