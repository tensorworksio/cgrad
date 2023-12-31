#include "tensor.h"
#include <stdbool.h>

int get_size(int shape[], int ndim);
bool same_shape(tensor_t* a, tensor_t* b);
void tensor_print_data(tensor_t* tensor);
void tensor_print_grad(tensor_t* tensor);
void tensor_print(tensor_t* tensor);