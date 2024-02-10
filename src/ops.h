#include <math.h>
#include "tensor.h"
#include "backops.h"
#include "helpers.h"

// UNARY OPS
float* relut(tensor_t *a, tensor_t *b);

// BINARY OPS
float* addt(tensor_t *a, tensor_t *b);
float* mult(tensor_t *a, tensor_t *b);
float* powt(tensor_t *a, tensor_t *b);

// REDUCE OPS
float* sumt(tensor_t *a, tensor_t *b);