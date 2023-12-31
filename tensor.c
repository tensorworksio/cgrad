#include "tensor.h"
#include "backops.h"
#include "helpers.h"

tensor_t* tensor_init(int size)
{   
    tensor_t* tensor = (tensor_t*)malloc(sizeof(tensor_t));
    tensor->data = (float*)malloc(sizeof(float) * size);
    tensor->grad = NULL; // grad is only allocated if needed
    tensor->size = size;
    tensor->child1 = NULL;
    tensor->child2 = NULL;
    tensor->backward = NULL;
    return tensor;
}

tensor_t* tensor_create(int shape[], int ndim, bool requires_grad)
{   
    int size = get_size(shape, ndim);
    tensor_t* tensor = tensor_init(size);
    tensor->ndim = ndim;
    tensor->requires_grad = requires_grad;
    tensor->shape = (int*)malloc(sizeof(int) * ndim);
    memcpy(tensor->shape, shape, sizeof(int) * ndim);
    if (requires_grad)
    {
        tensor->grad = (float*)malloc(sizeof(float) * size);
        tensor_set_grad(tensor, 0.0);
    }
    return tensor;
}

tensor_t* tensor_create_random(int shape[], int ndim, bool requires_grad)
{
    tensor_t* tensor = tensor_create(shape, ndim, requires_grad);
    for (int i = 0; i < tensor->size; ++i)
    {
        tensor->data[i] = (float)rand() / (float)RAND_MAX;
    }
    return tensor;
}

void tensor_free(tensor_t* tensor)
{
    free(tensor->data);
    free(tensor->grad);
    free(tensor->shape);
    free(tensor);
}

void tensor_set_data(tensor_t* tensor, float data)
{
    for (int i = 0; i < tensor->size; i++)
    {
        tensor->data[i] = data;
    }
}

void tensor_set_grad(tensor_t* tensor, float grad) 
{
    for (int i = 0; i < tensor->size; i++)
    {
        tensor->grad[i] = grad;
    }
}

void backward(tensor_t* tensor)
{   
    assert(tensor->requires_grad && "Cannot perform backward on tensor that has no grad.");
    assert(tensor->size == 1 && "Backward operation only supported for scalar tensors.");
    tensor_set_grad(tensor, 1.0);
    _backward(tensor);
}