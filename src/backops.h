#include <math.h>
#include "log.h"
#include "tensor.h"
#include "helpers.h"

void backward(tensor_t *self);
void init_grad(tensor_t *self);

// BACKWARD
void update_grad_relu(tensor_t *self, tensor_t *parent);
void update_grad_add(tensor_t *self, tensor_t *parent);
void update_grad_mul(tensor_t *self, tensor_t *parent, tensor_t *other);
void update_grad_pow(tensor_t *self, tensor_t *parent, tensor_t *other);
void update_grad_exp(tensor_t *self, tensor_t *parent, tensor_t *other);
void update_grad_sum(tensor_t *self, tensor_t *parent);

// UNARY OPS
void backward_relu(tensor_t *self);

// BINARY OPS
void backward_add(tensor_t *self);
void backward_mul(tensor_t *self);
void backward_pow(tensor_t *self);

// REDUCE OPS
void backward_sum(tensor_t *self);

// MOVEMENT OPS
void backward_ref(tensor_t *self);
void backward_slice(tensor_t *self);
void backward_copy(tensor_t *self);
