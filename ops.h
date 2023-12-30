#include <math.h>
#include "tensor.h"

// BINARY OPS
tensor_t* add(tensor_t* a, tensor_t* b);
tensor_t* mul(tensor_t* a, tensor_t* b);
tensor_t* power(tensor_t* a, tensor_t* b);

// REDUCE OPS
tensor_t* sum(tensor_t* a);