#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

typedef struct
{
    bool requires_grad;
    int* shape;
    int ndim;
} tensor_params_t;

typedef struct tensor
{   
    float* data;
    float* grad;
    
    int size;
    int ndim;
    int *shape;
    bool requires_grad;

    struct tensor* child1;
    struct tensor* child2;
    void (*backward) (struct tensor* self);
} tensor_t;


tensor_t* tensor_alloc(int size);
tensor_t* tensor_create(int shape[], int ndim, bool requires_grad);

tensor_t* tensor(const float data[], int shape[], int ndim, bool requires_grad);
tensor_t* tensor_rand(int shape[], int ndim, bool requires_grad);
tensor_t* tensor_zeros(int shape[], int ndim, bool requires_grad);
tensor_t* tensor_ones(int shape[], int ndim, bool requires_grad);

void tensor_zero_grad(tensor_t* tensor);
void tensor_init_grad(tensor_t* tensor);

bool tensor_same_shape(tensor_t* a, tensor_t* b);
bool tensor_equals(tensor_t* a, tensor_t* b, bool with_grad);

void tensor_free(tensor_t* tensor, bool recursive);
void tensor_set_data(tensor_t* self, float data[], int size);
void tensor_set_grad(tensor_t* self, float grad[], int size);
void tensor_print(tensor_t* tensor);

void tensor_backward(tensor_t* self);

#endif