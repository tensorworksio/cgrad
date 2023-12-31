#include "helpers.h"

int get_size(int shape[], int ndim)
{
    int size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size *= shape[i];
    }
    return size;
}

bool same_shape(tensor_t* a, tensor_t* b)
{
    if (a->ndim != b->ndim)
    {
        return false;
    }
    for (int i = 0; i < a->ndim; i++)
    {
        if (a->shape[i] != b->shape[i])
        {
            return false;
        }
    }
    return true;
}

void tensor_print_data(tensor_t* tensor)
{
    printf("data :: ");
    for (int i = 0; i < tensor->size; i++)
    {
        printf("%f ", tensor->data[i]);
    }
    printf("\n");
}

void tensor_print_grad(tensor_t* tensor)
{
    printf("grad :: ");
    for (int i = 0; i < tensor->size; i++)
    {
        printf("%f ", tensor->grad[i]);
    }
    printf("\n");
}

void tensor_print(tensor_t* tensor)
{   
    tensor_print_data(tensor);
    if (tensor->requires_grad) tensor_print_grad(tensor);
}