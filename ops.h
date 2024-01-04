#include <math.h>
#include "tensor.h"
#include "backops.h"
#include "helpers.h"

#define E 2.718281828459045

// UNARY OPS
tensor_t* tensor_neg(tensor_t* a);
tensor_t* tensor_exp(tensor_t* a);
tensor_t* tensor_relu(tensor_t* a);

// BINARY OPS
tensor_t* tensor_add(tensor_t* a, tensor_t* b);
tensor_t* tensor_add_tt(tensor_t* a, tensor_t* b);
tensor_t* tensor_add_tf(tensor_t* a, float b);
tensor_t* tensor_add_ft(float a, tensor_t* b);

tensor_t* tensor_mul(tensor_t* a, tensor_t* b);
tensor_t* tensor_mul_tt(tensor_t* a, tensor_t* b);
tensor_t* tensor_mul_tf(tensor_t* a, float b);
tensor_t* tensor_mul_ft(float a, tensor_t* b);

tensor_t* tensor_pow(tensor_t* a, tensor_t* b);
tensor_t* tensor_pow_tt(tensor_t* a, tensor_t* b);
tensor_t* tensor_pow_tf(tensor_t* a, float b);
tensor_t* tensor_pow_ft(float a, tensor_t* b);

tensor_t* tensor_sub(tensor_t* a, tensor_t* b);
tensor_t* tensor_sub_tt(tensor_t* a, tensor_t* b);
tensor_t* tensor_sub_tf(tensor_t* a, float b);
tensor_t* tensor_sub_ft(float a, tensor_t* b);

tensor_t* tensor_div(tensor_t* a, tensor_t* b);
tensor_t* tensor_div_tt(tensor_t* a, tensor_t* b);
tensor_t* tensor_div_tf(tensor_t* a, float b);
tensor_t* tensor_div_ft(float a, tensor_t* b);

// REDUCE OPS
tensor_t* tensor_sum(tensor_t* a);