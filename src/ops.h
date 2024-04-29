#include <math.h>
#include "tensor.h"
#include "backops.h"
#include "helpers.h"

void init_data(tensor_t *self);
void free_data(tensor_t *self);

// FORWARD
void relut(tensor_t *self, tensor_t *parent);
void addt(tensor_t *self, tensor_t *parent, tensor_t *other);
void mult(tensor_t *self, tensor_t *parent, tensor_t *other);
void powt(tensor_t *self, tensor_t *parent, tensor_t *other);
void catt(tensor_t *self, tensor_t *parents[], int n_parents);

// UNARY OPS
void forward_relu(tensor_t *self);

// BINARY OPS
void forward_add(tensor_t *self);
void forward_mul(tensor_t *self);
void forward_pow(tensor_t *self);

// MOVEMENT OPS
void forward_cat(tensor_t *self);
void forward_ref(tensor_t *self);
void forward_copy(tensor_t *self);