#include "backops.h"

void backward(tensor_t* self)
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

    int k;
    for (int i = 0; i < self->size; i++)
    {   
        k = (other->size == 1) ? 0 : i;
        child->grad[i] += self->grad[i] * other->data[k];
    }
}

void update_grad_pow(tensor_t* self, tensor_t* child, tensor_t* other)
{
    if (!child->requires_grad) return;

    int k;
    for (int i = 0; i < self->size; i++)
    {   
        k = (other->size == 1) ? 0 : i;
        child->grad[i] += self->grad[i] * other->data[k] * powf(child->data[i], other->data[k] - 1);
    }
}

void update_grad_exp(tensor_t* self, tensor_t* child, tensor_t* other)
{
    if (!child->requires_grad) return;

    int k;
    for (int i = 0; i < self->size; i++)
    {   
        k = (other->size == 1) ? 0 : i;
        child->grad[i] += self->grad[i] * self->data[i] * logf(other->data[k]);
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

    backward(self->child1);
    backward(self->child2);
}

void backward_mul(tensor_t* self)
{   
    if (self->child1 == NULL || self->child2 == NULL) {
        printf("A child is NULL\n");
        return;
    }
    update_grad_mul(self, self->child1, self->child2);
    update_grad_mul(self, self->child2, self->child1);

    backward(self->child1);
    backward(self->child2);
}

void backward_pow(tensor_t* self)
{   
    if (self->child1 == NULL) {
        printf("A child is NULL\n");
        return;
    }
    update_grad_pow(self, self->child1, self->child2);
    update_grad_exp(self, self->child2, self->child1);

    backward(self->child1);
    backward(self->child2);
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
    backward(self->child1);
}