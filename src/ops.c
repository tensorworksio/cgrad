#include "ops.h"

// FORWARD
float *forward_relu(int n_children, tensor_t **children)
{
    ASSERT(n_children == 1, "Relu forward must have 1 child, got %d", n_children);
    return relut(children[0]);
}

float *forward_add(int n_children, tensor_t **children)
{
    ASSERT(n_children == 2, "Add forward must have 2 children, got %d", n_children);
    return addt(children[0], children[1]);
}

float *forward_mul(int n_children, tensor_t **children)
{
    ASSERT(n_children == 2, "Mul forward must have 2 children, got %d", n_children);
    return mult(children[0], children[1]);
}

float *forward_pow(int n_children, tensor_t **children)
{
    ASSERT(n_children == 2, "Pow forward must have 2 children, got %d", n_children);
    return powt(children[0], children[1]);
}

float *forward_sum(int n_children, tensor_t **children)
{
    ASSERT(n_children == 1, "Sum forward must have 1 child, got %d", n_children);
    return sumt(children[0]);
}

// TODO:
// before any operation, check if full range is used
// if so, iterate over data contiguous memory
// if not, use iterator to iterate over data

// UNARY OPS
float *relut(tensor_t *a)
{
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
float *sumt(tensor_t *a)
{
    float *data = smalloc(.size = 1, .nmemb = sizeof(float), .kind = SHARED);
    for (int i = 0; i < a->size; i++)
    {
        data[0] += a->data[i];
    }
    return data;
}