#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
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
    struct tensor* child1;
    struct tensor* child2;
    void (*backward) (struct tensor* self);
} tensor_t;


tensor_t* tensor_create(int size);
tensor_t* tensor_create_random(int size);

void tensor_free(tensor_t* tensor);
void tensor_set_data(tensor_t* self, float data);
void tensor_set_grad(tensor_t* self, float grad);
void tensor_print(tensor_t* tensor);
void backward(tensor_t* self);

#endif