#include <math.h>
#include "tensor.h"

void _backward(tensor_t* self);

// BACKWARD OPS
void backward_add(tensor_t* self);
void backward_mul(tensor_t* self);
void backward_power(tensor_t* self);

// REDUCE OPS
void backward_sum(tensor_t* self);