#include "tensor.h"
#include <stdbool.h>

int get_size(int shape[], int ndim);
bool same_shape(tensor_t* a, tensor_t* b);
void print_tensor(float* data, int shape[], int ndim);