#ifndef TENSOR_H
#define TENSOR_H

#include "slice.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>
#include <csptr/smart_ptr.h>

typedef enum {
    PRINT_SHAPE = 0,
    PRINT_STRIDE = (1 << 0),
    PRINT_DATA = (1 << 1),
    PRINT_GRAD = (1 << 2),
    PRINT_CHILDREN = (1 << 3),
    PRINT_ALL = PRINT_SHAPE | PRINT_STRIDE | PRINT_DATA | PRINT_GRAD | PRINT_CHILDREN
} flag_t;

typedef struct tensor
{
    float *data;
    float *grad;
    bool requires_grad;

    int size;
    int ndim;
    int *shape;
    int *stride;

    struct tensor *child1;
    struct tensor *child2;
    void (*backward)(struct tensor *self);
} tensor_t;

tensor_t *tensor_alloc(int size);
tensor_t *tensor_create(int shape[], int ndim, bool requires_grad);
tensor_t *tensor_init(int shape[], int ndim, bool requires_grad);

tensor_t *tensor(const float data[], int shape[], int ndim, bool requires_grad);
tensor_t *tensor_rand(int shape[], int ndim, bool requires_grad);
tensor_t *tensor_zeros(int shape[], int ndim, bool requires_grad);
tensor_t *tensor_ones(int shape[], int ndim, bool requires_grad);

tensor_t *tensor_reshape(tensor_t *tensor, int shape[], int ndim);
tensor_t *tensor_transpose(tensor_t *self, int axis1, int axis2);
tensor_t *tensor_slice(tensor_t *self, slice_t ranges[]);
tensor_t *tensor_cat(tensor_t *tensors[], int n_tensors, int axis);

void tensor_copy(tensor_t *dst, tensor_t *src, int *dst_idx, int *src_idx, slice_t *ranges, int dim);

void tensor_zero_grad(tensor_t *tensor);
void tensor_init_grad(tensor_t *tensor);

bool tensor_same_shape(tensor_t *a, tensor_t *b, bool debug);
bool tensor_equals(tensor_t *a, tensor_t *b, bool with_grad);

void tensor_free(tensor_t *tensor, bool recursive);
void tensor_set_data(tensor_t *self, float data[], int size);
void tensor_set_grad(tensor_t *self, float grad[], int size);

void tensor_print(tensor_t *tensor, flag_t flags);

void tensor_backward(tensor_t *self);

#endif