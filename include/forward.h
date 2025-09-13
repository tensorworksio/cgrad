#include "backward.h"
#include "helpers.h"
#include "iterator.h"
#include "tensor.h"
#include <math.h>

// FORWARD
void relut (tensor_t *self, tensor_t *child);
void addt (tensor_t *self, tensor_t *child, tensor_t *other);
void mult (tensor_t *self, tensor_t *child, tensor_t *other);
void powt (tensor_t *self, tensor_t *child, tensor_t *other);
void sumt (tensor_t *self, tensor_t *child);

// UNARY OPS
void forward_relu (tensor_t *self);

// BINARY OPS
void forward_add (tensor_t *self);
void forward_mul (tensor_t *self);
void forward_pow (tensor_t *self);

// REDUCE OPS
void forward_sum (tensor_t *self);

// MOVEMENT OPS
void forward_ref (tensor_t *self);
void forward_copy (tensor_t *self);
void forward_slice (tensor_t *self);
void forward_cat (tensor_t *self);