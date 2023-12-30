#include "tensor.h"
#include "backops.h"

tensor_t* tensor_create(int size)
{   
    tensor_t* tensor = (tensor_t*)malloc(sizeof(tensor_t));
    tensor->data = (float*)malloc(sizeof(float) * size);
    tensor->grad = (float*)malloc(sizeof(float) * size);
    tensor->size = size;
    tensor->child1 = NULL;
    tensor->child2 = NULL;
    tensor->backward = NULL;
    return tensor;
}

tensor_t* tensor_create_random(int size)
{
    tensor_t* tensor = tensor_create(size);
    for (int i = 0; i < size; ++i)
    {
        tensor->data[i] = (float)rand() / (float)RAND_MAX;
    }
    return tensor;
}

void tensor_free(tensor_t* tensor)
{
    free(tensor->data);
    free(tensor->grad);
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
    assert(tensor->backward != NULL);
    assert(tensor->size == 1 && "Backward operation only supported for scalar tensors.");
    tensor_set_grad(tensor, 1.0);
    _backward(tensor);
}

void tensor_print(tensor_t* tensor)
{   
    printf("data :: ");
    for (int i = 0; i < tensor->size; i++)
    {
        printf("%f ", tensor->data[i]);
    }
    printf("\n");
    printf("grad :: ");
    for (int i = 0; i < tensor->size; i++)
    {
        printf("%f ", tensor->grad[i]);
    }
    printf("\n");
}