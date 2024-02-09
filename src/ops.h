#include <math.h>
#include "tensor.h"
#include "backops.h"
#include "helpers.h"

// UNARY OPS
float* negt(tensor_t *a);
float* expt(tensor_t *a);
float* relut(tensor_t *a);

// BINARY OPS
float* addt(tensor_t *a, tensor_t *b);
float* mult(tensor_t *a, tensor_t *b);
float* powt(tensor_t *a, tensor_t *b);

float* subt(tensor_t *a, tensor_t *b);
float* divt(tensor_t *a, tensor_t *b);

// REDUCE OPS
float* sumt(tensor_t *a);