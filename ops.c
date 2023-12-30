#include "tensor.h"
#include "backops.h"
#include "ops.h"

tensor_t* add(tensor_t* a,  tensor_t* b)
{   
    assert(a->size == b->size && "Size mismatch");
    tensor_t* out = tensor_create(a->size);
    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }
    out->child1 = a;
    out->child2 = b;
    out->backward = backward_add;

    return out;
}

tensor_t* mul(tensor_t* a,  tensor_t* b)
{   
    assert(a->size == b->size && "Size mismatch");
    tensor_t* out = tensor_create(a->size);
    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = a->data[i] * b->data[i];
    }
    out->child1 = a;
    out->child2 = b;
    out->backward = backward_mul;

    return out;
}

tensor_t* power(tensor_t* a, tensor_t* b)
{   
    assert(a->size == b->size && "Size mismatch");
    tensor_t* out = tensor_create(a->size);
    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = powf(a->data[i], b->data[i]);
    }
    out->child1 = a;
    out->child2 = b;
    out->backward = backward_power;

    return out;
}

tensor_t* sum(tensor_t* a)
{   
    tensor_t* out = tensor_create(1);
    for (int i = 0; i < a->size; i++)
    {
        out->data[0] += a->data[i];
    }
    out->child1 = a;
    out->backward = backward_sum;

    return out;
}