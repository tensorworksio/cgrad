#include <math.h>
#include "tensor.h"
#include "backops.h"
#include "helpers.h"

void init_data(tensor_t *self);
void free_data(tensor_t *self);

// FORWARD
void forward_relu(tensor_t *self);
void forward_add(tensor_t *self);
void forward_mul(tensor_t *self);
void forward_pow(tensor_t *self);
void forward_sum(tensor_t *self);
void forward_slice(tensor_t *self);
void forward_cat(tensor_t *self);
void forward_nop(tensor_t *self);

// UNARY OPS
void relut(tensor_t *self, tensor_t *child);

// BINARY OPS
void addt(tensor_t *self, tensor_t *child, tensor_t *other);
void mult(tensor_t *self, tensor_t *child, tensor_t *other);
void powt(tensor_t *self, tensor_t *child, tensor_t *other);

// REDUCE OPS
void sumt(tensor_t *self, tensor_t *child);

// MOVEMENT OPS
void catt(tensor_t *self, tensor_t *children[], int n_children);