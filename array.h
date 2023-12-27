#ifndef ARRAY_H
#define ARRAY_H

#include <stdio.h>
#include <stdlib.h>

typedef struct array
{   
    float* data;
    float* grad;
    int size;
    struct array* child;
    void (*backward) (struct array* self);
} array_t;


array_t* array_create(int size);
array_t* array_create_random(int size);
void array_free(array_t* array);
void array_zero_grad(array_t* array);
void array_print(array_t* array);

#endif