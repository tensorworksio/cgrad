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

tensor_t* tensor(const float data[], int shape[], int ndim, bool requires_grad)
{
    tensor_t* tensor = tensor_create(shape, ndim, requires_grad);
    memcpy(tensor->data, data, sizeof(float) * tensor->size);
    return tensor;
}

tensor_t* tensor_rand(int shape[], int ndim, bool requires_grad)
{
    tensor_t* tensor = tensor_create(shape, ndim, requires_grad);
    for (int i = 0; i < tensor->size; ++i)
    {
        tensor->data[i] = (float)rand() / (float)RAND_MAX;
    }
    return tensor;
}

tensor_t* tensor_zeros(int shape[], int ndim, bool requires_grad)
{
    tensor_t* tensor = tensor_create(shape, ndim, requires_grad);
    tensor_set_data(tensor, 0.0);
    return tensor;
}

tensor_t* tensor_ones(int shape[], int ndim, bool requires_grad)
{
    tensor_t* tensor = tensor_create(shape, ndim, requires_grad);
    tensor_set_data(tensor, 1.0);
    return tensor;
}

tensor_t* tensor_const(int shape[], int ndim, bool requires_grad, float value)
{
    tensor_t* tensor = tensor_create(shape, ndim, requires_grad);
    tensor_set_data(tensor, value);
    return tensor;
}

void tensor_free(tensor_t* tensor, bool recursive)
{
    if (recursive)
    {
        if (tensor->child1)
        {
            tensor_free(tensor->child1, recursive);
        }
        if (tensor->child2)
        {
            tensor_free(tensor->child2, recursive);
        }
    }

    free(tensor->data);
    free(tensor->grad);
    free(tensor->shape);
    free(tensor);
}

void tensor_print(tensor_t* tensor)
{   
    printf("DATA\n");
    print_tensor_data(tensor->data, tensor->shape, tensor->ndim);
    printf("\n");
    if (tensor->requires_grad) {
        printf("GRAD\n");
        print_tensor_data(tensor->grad, tensor->shape, tensor->ndim);
        printf("\n");
    }
}

void tensor_set_data(tensor_t* tensor, float value)
{
    set_tensor_data(tensor->data, tensor->size, value);
}

void tensor_set_grad(tensor_t* tensor, float value) 
{
    set_tensor_data(tensor->grad, tensor->size, value);
}

void tensor_backward(tensor_t* tensor)
{   
    if (!tensor->requires_grad) {
        fprintf(stderr, "Cannot perform backward on tensor that has no grad.\n");
        tensor_free(tensor, true);
        exit(EXIT_FAILURE);
    }
    if (tensor->size != 1) {
        fprintf(stderr, "Backward operation only supported for scalar tensors.\n");
        tensor_free(tensor, true);
        exit(EXIT_FAILURE);
    }
    tensor_set_grad(tensor, 1.0);
    backward(tensor);
}