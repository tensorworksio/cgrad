#include <math.h>
#include "log.h"
#include "tensor.h"
#include "helpers.h"

void backward(tensor_t* self);

// UNARY OPS
void backward_relu(tensor_t* self);
void update_grad_relu(tensor_t* self, tensor_t* child);

// BINARY OPS
void backward_add(tensor_t* self);
void update_grad_add(tensor_t* self, tensor_t* child);

void backward_mul(tensor_t* self);
void update_grad_mul(tensor_t* self, tensor_t* child, tensor_t* other);

void backward_pow(tensor_t* self);
void update_grad_pow(tensor_t* self, tensor_t* child, tensor_t* other);
void update_grad_exp(tensor_t* self, tensor_t* child, tensor_t* other);

// REDUCE OPS
void backward_sum(tensor_t* self);