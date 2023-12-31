#include "backops.h"

void _backward(tensor_t* self)
{
    if (self->backward == NULL) {
        printf("No backward function implemented for that operator\n");
        return;
    }
    self->backward(self);
}

void backward_add(tensor_t* self)
{   
    if (self->child1 == NULL || self->child2 == NULL) {
        printf("A child is NULL\n");
        return;
    }
    for (int i = 0; i < self->size; i++)
    {   
        self->child1->grad[i] += self->grad[i];
        self->child2->grad[i] += self->grad[i];
        
    }
    _backward(self->child1);
    _backward(self->child2);
}

void backward_mul(tensor_t* self)
{   
    if (self->child1 == NULL || self->child2 == NULL) {
        printf("A child is NULL\n");
        return;
    }
    for (int i = 0; i < self->size; i++)
    {   
        self->child1->grad[i] += self->grad[i] * self->child2->data[i];
        self->child2->grad[i] += self->grad[i] * self->child1->data[i];
    }
    _backward(self->child1);
    _backward(self->child2);
}

void backward_power(tensor_t* self)
{   
    if (self->child1 == NULL) {
        printf("A child is NULL\n");
        return;
    }
    for (int i = 0; i < self->size; i++)
    {   
        self->child1->grad[i] += self->grad[i] * self->child2->data[i] * powf(self->child1->data[i], self->child2->data[i] - 1);
    }
    _backward(self->child1);
}

void backward_sum(tensor_t* self)
{   
    if (self->child1 == NULL) {
        printf("A child is NULL\n");
        return;
    }
    for (int i = 0; i < self->child1->size; i++)
    {   
        self->child1->grad[i] += self->grad[0];
    }
    _backward(self->child1);
}