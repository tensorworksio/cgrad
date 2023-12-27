#include "array.h"
#include "backops.h"
#include "ops.h"

array_t* add(array_t* a,  array_t* b)
{
    array_t* out = array_create(a->size);
    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }
    out->child1 = a;
    out->child2 = b;
    out->backward = backward_add;

    return out;
}

array_t* mul(array_t* a,  array_t* b)
{
    array_t* out = array_create(a->size);
    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = a->data[i] * b->data[i];
    }
    out->child1 = a;
    out->child2 = b;
    out->backward = backward_mul;

    return out;
}

