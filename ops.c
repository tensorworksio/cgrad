#include "tensor.h"
#include "backops.h"
#include "helpers.h"
#include "ops.h"


tensor_t* add(tensor_t* a,  tensor_t* b)
{   
    assert(same_shape(a, b) && "Shape mismatch");
    tensor_t* out = tensor_create(a->shape, a->ndim, a->requires_grad || b->requires_grad);
    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }
    out->child1 = a;
    out->child2 = b;
    if (out->requires_grad) out->backward = backward_add;

    return out;
}

tensor_t* mul(tensor_t* a,  tensor_t* b)
{   
    assert(same_shape(a, b) && "Shape mismatch");
    tensor_t* out = tensor_create(a->shape, a->ndim, a->requires_grad || b->requires_grad);
    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = a->data[i] * b->data[i];
    }
    out->child1 = a;
    out->child2 = b;
    if (out->requires_grad) out->backward = backward_mul;

    return out;
}

tensor_t* power(tensor_t* a, tensor_t* b)
{   
    assert(same_shape(a, b) && "Shape mismatch");
    tensor_t* out = tensor_create(a->shape, a->ndim, a->requires_grad || b->requires_grad);
    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = powf(a->data[i], b->data[i]);
    }
    out->child1 = a;
    out->child2 = b;
    if (out->requires_grad) out->backward = backward_power;

    return out;
}

tensor_t* sum(tensor_t* a)
{   
    tensor_t* out = tensor_create((int[]){1}, 1, a->requires_grad);
    for (int i = 0; i < a->size; i++)
    {
        out->data[0] += a->data[i];
    }
    out->child1 = a;
    if (out->requires_grad) out->backward = backward_sum;

    return out;
}