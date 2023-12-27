#include "array.h"
#include "backops.h"

void backward_add(array_t* self)
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
    backward(self->child1);
    backward(self->child2);
}

void backward_mul(array_t* self)
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
    backward(self->child1);
    backward(self->child2);
}