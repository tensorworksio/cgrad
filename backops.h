#include <math.h>
#include "tensor.h"
#include "logger.h"
#include "helpers.h"

void _backward(tensor_t* self);

// BINARY OPS
void backward_add(tensor_t* self);
void update_grad_add(tensor_t* self, tensor_t* child);

void backward_mul(tensor_t* self);
void update_grad_mul(tensor_t* self, tensor_t* child, tensor_t* other);

void backward_pow(tensor_t* self);

// REDUCE OPS
void backward_sum(tensor_t* self);