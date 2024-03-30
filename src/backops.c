#include "backops.h"

void backward(tensor_t *self)
{
    if (self->backward == NULL)
    {
        log_debug("Node %p has no backward function.", (void *)self);
        return;
    }
    if (self->children == NULL)
    {
        log_debug("Node %p has no children.", (void *)self);
        return;
    }
    self->backward(self);
}

void init_grad(tensor_t *self)
{
    if (!self->requires_grad || self->grad)
        return;
    self->grad = smalloc(.size = self->size, .nmemb = sizeof(float), .kind = SHARED);
    tensor_zero_grad(self);
}

// BACKWARD
void update_grad_add(tensor_t *self, tensor_t *child)
{
    if (!child->requires_grad)
        return;

    for (int i = 0; i < self->size; i++)
    {
        child->grad[i] += self->grad[i];
    }
}

void update_grad_relu(tensor_t *self, tensor_t *child)
{
    if (!child->requires_grad)
        return;
    for (int i = 0; i < self->size; i++)
    {
        child->grad[i] += self->grad[i] * (self->data[i] > 0);
    }
}

void update_grad_mul(tensor_t *self, tensor_t *child, tensor_t *other)
{
    if (!child->requires_grad)
        return;

    int k;
    for (int i = 0; i < self->size; i++)
    {
        k = (other->size == 1) ? 0 : i;
        child->grad[i] += self->grad[i] * other->data[k];
    }
}

void update_grad_pow(tensor_t *self, tensor_t *child, tensor_t *other)
{
    if (!child->requires_grad)
        return;

    int k;
    for (int i = 0; i < self->size; i++)
    {
        k = (other->size == 1) ? 0 : i;
        child->grad[i] += self->grad[i] * other->data[k] * powf(child->data[i], other->data[k] - 1);
    }
}

void update_grad_exp(tensor_t *self, tensor_t *child, tensor_t *other)
{
    if (!child->requires_grad)
        return;

    int k;
    for (int i = 0; i < self->size; i++)
    {
        k = (other->size == 1) ? 0 : i;
        child->grad[i] += self->grad[i] * self->data[i] * logf(other->data[k]);
    }
}

void update_grad_sum(tensor_t *self, tensor_t *child)
{
    if (!child->requires_grad)
        return;
    for (int i = 0; i < child->size; i++)
    {
        child->grad[i] += self->grad[0];
    }
}

// UNARY OPS
void backward_relu(tensor_t *self)
{
    ASSERT(self->n_children == 1, "backward_relu expects 1 child, got %d", self->n_children);
    init_grad(self->children[0]);
    update_grad_relu(self, self->children[0]);
    backward(self->children[0]);
}

// BINARY OPS
void backward_add(tensor_t *self)
{
    ASSERT(self->n_children == 2, "backward_add expects 2 children, got %d", self->n_children);
    init_grad(self->children[0]);
    init_grad(self->children[1]);

    update_grad_add(self, self->children[0]);
    update_grad_add(self, self->children[1]);

    backward(self->children[0]);
    backward(self->children[1]);
}

void backward_mul(tensor_t *self)
{
    ASSERT(self->n_children == 2, "backward_mul expects 2 children, got %d", self->n_children);
    init_grad(self->children[0]);
    init_grad(self->children[1]);

    update_grad_mul(self, self->children[0], self->children[1]);
    update_grad_mul(self, self->children[1], self->children[0]);

    backward(self->children[0]);
    backward(self->children[1]);
}

void backward_pow(tensor_t *self)
{
    ASSERT(self->n_children == 2, "backward_pow expects 2 children, got %d", self->n_children);
    init_grad(self->children[0]);
    init_grad(self->children[1]);

    update_grad_pow(self, self->children[0], self->children[1]);
    update_grad_exp(self, self->children[1], self->children[0]);

    backward(self->children[0]);
    backward(self->children[1]);
}

// REDUCE OPS
void backward_sum(tensor_t *self)
{
    ASSERT(self->n_children == 1, "backward_sum expects 1 child, got %d", self->n_children);
    init_grad(self->children[0]);
    update_grad_sum(self, self->children[0]);
    backward(self->children[0]);
}

// MOVEMENT OPS
void backward_ref(tensor_t *self)
{
    ASSERT(self->n_children > 0, "backward_ref expects at least 1 child, got %d", self->n_children);
    for (int i = 0; i < self->n_children; i++)
    {
        self->children[i]->grad = sref(self->grad);
        backward(self->children[i]);
    }
}

void backward_slice(tensor_t *self)
{
    ASSERT(self->n_children == 1, "backward_slice expects 1 child, got %d", self->n_children);
    init_grad(self->children[0]);
    iterator_t it = tensor_iterator(self);
    copy_to_range(self->children[0]->grad, self->grad, &it);
    iterator_free(&it);
    sfree(self->grad);
    self->grad = sref(self->children[0]->grad);
    backward(self->children[0]);
}

void backward_copy(tensor_t *self)
{
    ASSERT(self->n_children == 1, "backward_copy expects 1 child, got %d", self->n_children);
    init_grad(self->children[0]);
    iterator_t it = tensor_iterator(self);
    copy_to_range(self->children[0]->grad, self->grad, &it);
    iterator_free(&it);
    backward(self->children[0]);
}