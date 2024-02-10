#include "ops.h"

// TODO:
// before any operation, check if full range is used
// if so, iterate over data contiguous memory
// if not, use iterator to iterate over data

// UNARY OPS
float *relut(tensor_t *a, tensor_t *b)
{   
    (void) b;
    int size = a->size;
    float *data = smalloc(.size = size, .nmemb = sizeof(float), .kind = SHARED);
    for (int i = 0; i < size; i++)
    {
        data[i] = (a->data[i] > 0.0) ? a->data[i] : 0.0;
    }
    return data;
}

// BINARY OPS
float *addt(tensor_t *a, tensor_t *b)
{   
    int j, k;
    int size = (a->size > b->size) ? a->size : b->size;
    float *data = smalloc(.size = size, .nmemb = sizeof(float), .kind = SHARED);
    for (int i = 0; i < size; i++)
    {   
        j = (a->size == 1) ? 0 : i;
        k = (b->size == 1) ? 0 : i;
        data[i] = a->data[j] + b->data[k];
    }
    return data;
}

float *mult(tensor_t *a, tensor_t *b)
{   
    int j, k;
    int size = (a->size > b->size) ? a->size : b->size;
    float *data = smalloc(.size = size, .nmemb = sizeof(float), .kind = SHARED);
    for (int i = 0; i < size; i++)
    {   
        j = (a->size == 1) ? 0 : i;
        k = (b->size == 1) ? 0 : i;
        data[i] = a->data[j] * b->data[k];
    }
    return data;
}

float *powt(tensor_t *a, tensor_t *b)
{   
    int j, k;
    int size = (a->size > b->size) ? a->size : b->size;
    float *data = smalloc(.size = size, .nmemb = sizeof(float), .kind = SHARED);
    for (int i = 0; i < size; i++)
    {   
        j = (a->size == 1) ? 0 : i;
        k = (b->size == 1) ? 0 : i;
        data[i] = powf(a->data[j], b->data[k]);
    }
    return data;
}

// REDUCE OPS
float *sumt(tensor_t *a, tensor_t *b)
{   
    (void) b;
    float *data = smalloc(.size = 1, .nmemb = sizeof(float), .kind = SHARED);
    for (int i = 0; i < a->size; i++)
    {
        data[0] += a->data[i];
    }
    return data;
}