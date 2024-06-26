#ifndef TENSOR_H
#define TENSOR_H

#include "assert.h"
#include "slice.h"
#include "iterator.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>
#include <csptr/smart_ptr.h>

typedef enum
{
    PRINT_SHAPE = 0,
    PRINT_STRIDE = (1 << 0),
    PRINT_DATA = (1 << 1),
    PRINT_GRAD = (1 << 2),
    PRINT_CHILDREN = (1 << 3),
    PRINT_ALL = PRINT_SHAPE | PRINT_STRIDE | PRINT_DATA | PRINT_GRAD | PRINT_CHILDREN
} print_flag_t;

typedef struct tensor
{
    float *data;
    float *grad;
    bool requires_grad;

    int size;
    int ndim;
    int *shape;
    int *stride;
    slice_t *range;

    int n_parents;
    struct tensor **parents;

    int n_children;
    struct tensor **children;

    void (*backward)(struct tensor *self);
    void (*forward)(struct tensor *self);

} tensor_t;

// ALLOC OPS
tensor_t *tensor_alloc(int size);
tensor_t *tensor_create(int shape[], int ndim, bool requires_grad);
tensor_t *tensor_init(int shape[], int ndim, bool requires_grad, void (*op)(tensor_t *));
void tensor_link(tensor_t *child, tensor_t *parent);

// DESTRUCT OPS
void tensor_unlink(tensor_t *child, tensor_t *parent);
void tensor_free(tensor_t *tensor, bool recursive);

// INIT OPS
tensor_t *tensor(const float data[], int shape[], int ndim, bool requires_grad);
tensor_t *tensor_copy(tensor_t *tensor, bool with_grad);
tensor_t *tensor_rand(int shape[], int ndim, bool requires_grad);
tensor_t *tensor_zeros(int shape[], int ndim, bool requires_grad);
tensor_t *tensor_ones(int shape[], int ndim, bool requires_grad);

void tensor_zero_grad(tensor_t *tensor);
void tensor_init_grad(tensor_t *tensor);
void tensor_set_data(tensor_t *self, float data[], int size);
void tensor_set_grad(tensor_t *self, float grad[], int size);

// COMPARISON OPS
bool tensor_same_shape(tensor_t *a, tensor_t *b, bool debug);
bool tensor_equals(tensor_t *a, tensor_t *b, bool with_grad);

// UNARY OPS
tensor_t *tensor_neg(tensor_t *tensor);
tensor_t *tensor_exp(tensor_t *tensor);
tensor_t *tensor_relu(tensor_t *tensor);

// BINARY OPS
tensor_t *tensor_add(tensor_t *a, tensor_t *b);
tensor_t *tensor_sub(tensor_t *a, tensor_t *b);
tensor_t *tensor_mul(tensor_t *a, tensor_t *b);
tensor_t *tensor_div(tensor_t *a, tensor_t *b);
tensor_t *tensor_pow(tensor_t *a, tensor_t *b);

tensor_t *tensor_add_tt(tensor_t *a, tensor_t *b);
tensor_t *tensor_add_tf(tensor_t *a, float b);
tensor_t *tensor_add_ft(float a, tensor_t *b);

tensor_t *tensor_sub_tt(tensor_t *a, tensor_t *b);
tensor_t *tensor_sub_tf(tensor_t *a, float b);
tensor_t *tensor_sub_ft(float a, tensor_t *b);

tensor_t *tensor_mul_tt(tensor_t *a, tensor_t *b);
tensor_t *tensor_mul_tf(tensor_t *a, float b);
tensor_t *tensor_mul_ft(float a, tensor_t *b);

tensor_t *tensor_div_tt(tensor_t *a, tensor_t *b);
tensor_t *tensor_div_tf(tensor_t *a, float b);
tensor_t *tensor_div_ft(float a, tensor_t *b);

tensor_t *tensor_pow_tt(tensor_t *a, tensor_t *b);
tensor_t *tensor_pow_tf(tensor_t *a, float b);
tensor_t *tensor_pow_ft(float a, tensor_t *b);

// REDUCE OPS
tensor_t *tensor_sum(tensor_t *tensor);
tensor_t *tensor_sum_axis(tensor_t *tensor, int axis);

// MOVEMENT OPS
tensor_t *tensor_reshape(tensor_t *tensor, int shape[], int ndim);
tensor_t *tensor_transpose(tensor_t *self, int axis1, int axis2);
tensor_t *tensor_slice(tensor_t *self, slice_t ranges[]);
tensor_t *tensor_cat(tensor_t *tensors[], int n_tensors, int axis);

// FORCING OPS
void tensor_forward(tensor_t *tensor);
void tensor_backward(tensor_t *tensor);
void tensor_print(tensor_t *tensor, print_flag_t flags);
iterator_t tensor_iterator(tensor_t *tensor);

#endif