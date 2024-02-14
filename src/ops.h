#include <math.h>
#include "tensor.h"
#include "backops.h"
#include "helpers.h"

// FORWARD
float *forward_relu(int n_children, tensor_t **children);
float *forward_add(int n_children, tensor_t **children);
float *forward_mul(int n_children, tensor_t **children);
float *forward_pow(int n_children, tensor_t **children);
float *forward_sum(int n_children, tensor_t **children);

// UNARY OPS
float *relut(tensor_t *a);

// BINARY OPS
float *addt(tensor_t *a, tensor_t *b);
float *mult(tensor_t *a, tensor_t *b);
float *powt(tensor_t *a, tensor_t *b);

// REDUCE OPS
float *sumt(tensor_t *a);