#include "backops.h"

void _backward(tensor_t* self)
{
    if (self->backward == NULL) {
        log_debug("No backward function implemented for that node.\n");
        return;
    }
    self->backward(self);
}

void update_grad_add(tensor_t* self, tensor_t* child)
{
    if (!child->requires_grad) return;
    for (int i = 0; i < self->size; i++)
    {   
        child->grad[i] += self->grad[i];
    }
}

void update_grad_mul(tensor_t* self, tensor_t* child, tensor_t* other)
{
    if (!child->requires_grad) return;
    if (other->size == 1) {
        for (int i = 0; i < self->size; i++)
        {   
            child->grad[i] += self->grad[i] * other->data[0];
        }
        return;
    }
    for (int i = 0; i < self->size; i++)
    {   
        child->grad[i] += self->grad[i] * other->data[i];
    }
}

void backward_add(tensor_t* self)
{   
    if (self->child1 == NULL || self->child2 == NULL) {
        printf("A child is NULL\n");
        return;
    }
    update_grad_add(self, self->child1);
    update_grad_add(self, self->child2);

    _backward(self->child1);
    _backward(self->child2);
}

void backward_mul(tensor_t* self)
{   
    if (self->child1 == NULL || self->child2 == NULL) {
        printf("A child is NULL\n");
        return;
    }
    update_grad_mul(self, self->child1, self->child2);
    update_grad_mul(self, self->child2, self->child1);

    _backward(self->child1);
    _backward(self->child2);
}

void backward_pow(tensor_t* self)
{   
    if (self->child1 == NULL) {
        printf("A child is NULL\n");
        return;
    }
    for (int i = 0; i < self->size; i++)
    {   
        if (self->child1->requires_grad) {
            self->child1->grad[i] += self->grad[i] * self->child2->data[i] * powf(self->child1->data[i], self->child2->data[i] - 1);
        }
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
        if (self->child1->requires_grad) self->child1->grad[i] += self->grad[0];
    }
    _backward(self->child1);
}