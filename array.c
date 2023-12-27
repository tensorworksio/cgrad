#include "array.h"

array_t* array_create(int size)
{   
    array_t* array = (array_t*)malloc(sizeof(array_t));
    array->data = (float*)malloc(sizeof(float) * size);
    array->grad = (float*)malloc(sizeof(float) * size);
    array_zero_grad(array);
    array->size = size;
    array->child1 = NULL;
    array->child2 = NULL;
    array->backward = NULL;
    return array;
}

array_t* array_create_random(int size)
{
    array_t* array = array_create(size);
    for (int i = 0; i < size; i++)
    {
        array->data[i] = (float)rand() / (float)RAND_MAX;
    }
    array_zero_grad(array);
    return array;
}


void array_free(array_t* array)
{
    free(array->data);
    free(array->grad);
    free(array);
}

void array_zero_grad(array_t* array)
{
    for (int i = 0; i < array->size; i++)
    {
        array->grad[i] = 0;
    }
}

void array_print(array_t* array)
{   
    printf("data :: ");
    for (int i = 0; i < array->size; i++)
    {
        printf("%f ", array->data[i]);
    }
    printf("\n");
    printf("grad :: ");
    for (int i = 0; i < array->size; i++)
    {
        printf("%f ", array->grad[i]);
    }
    printf("\n");
}

void backward(array_t* self)
{
    if (self->backward == NULL) {
        printf("No backward function\n");
        return;
    }
    self->backward(self);
}