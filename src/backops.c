#include "backops.h"

void backward(tensor_t *self)
{
    if (self->backward == NULL)
    {
        log_debug("Node %p has no backward function.", (void *)self);
        return;
    }
    if (self->parents == NULL)
    {
        log_debug("Node %p has no parents.", (void *)self);
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
void update_grad_add(tensor_t *self, tensor_t *parent)
{
    if (!parent->requires_grad)
        return;

    for (int i = 0; i < self->size; i++)
    {
        parent->grad[i] += self->grad[i];
    }
}

void update_grad_relu(tensor_t *self, tensor_t *parent)
{
    if (!parent->requires_grad)
        return;
    for (int i = 0; i < self->size; i++)
    {
        parent->grad[i] += self->grad[i] * (self->data[i] > 0);
    }
}

void update_grad_mul(tensor_t *self, tensor_t *parent, tensor_t *other)
{
    if (!parent->requires_grad)
        return;

    int k;
    for (int i = 0; i < self->size; i++)
    {
        k = (other->size == 1) ? 0 : i;
        parent->grad[i] += self->grad[i] * other->data[k];
    }
}

void update_grad_pow(tensor_t *self, tensor_t *parent, tensor_t *other)
{
    if (!parent->requires_grad)
        return;

    int k;
    for (int i = 0; i < self->size; i++)
    {
        k = (other->size == 1) ? 0 : i;
        parent->grad[i] += self->grad[i] * other->data[k] * powf(parent->data[i], other->data[k] - 1);
    }
}

void update_grad_exp(tensor_t *self, tensor_t *parent, tensor_t *other)
{
    if (!parent->requires_grad)
        return;

    int k;
    for (int i = 0; i < self->size; i++)
    {
        k = (other->size == 1) ? 0 : i;
        parent->grad[i] += self->grad[i] * self->data[i] * logf(other->data[k]);
    }
}

// UNARY OPS
void backward_relu(tensor_t *self)
{
    ASSERT(self->n_parents == 1, "backward_relu expects 1 parent, got %d", self->n_parents);
    init_grad(self->parents[0]);
    update_grad_relu(self, self->parents[0]);
    backward(self->parents[0]);
}

// BINARY OPS
void backward_add(tensor_t *self)
{
    ASSERT(self->n_parents == 2, "backward_add expects 2 parents, got %d", self->n_parents);
    init_grad(self->parents[0]);
    init_grad(self->parents[1]);

    update_grad_add(self, self->parents[0]);
    update_grad_add(self, self->parents[1]);

    backward(self->parents[0]);
    backward(self->parents[1]);
}

void backward_mul(tensor_t *self)
{
    ASSERT(self->n_parents == 2, "backward_mul expects 2 parents, got %d", self->n_parents);
    init_grad(self->parents[0]);
    init_grad(self->parents[1]);

    update_grad_mul(self, self->parents[0], self->parents[1]);
    update_grad_mul(self, self->parents[1], self->parents[0]);

    backward(self->parents[0]);
    backward(self->parents[1]);
}

void backward_pow(tensor_t *self)
{
    ASSERT(self->n_parents == 2, "backward_pow expects 2 parents, got %d", self->n_parents);
    init_grad(self->parents[0]);
    init_grad(self->parents[1]);

    update_grad_pow(self, self->parents[0], self->parents[1]);
    update_grad_exp(self, self->parents[1], self->parents[0]);

    backward(self->parents[0]);
    backward(self->parents[1]);
}

// MOVEMENT OPS
void backward_ref(tensor_t *self)
{
    ASSERT(self->n_parents > 0, "backward_ref expects at least 1 parent, got %d", self->n_parents);
    for (int i = 0; i < self->n_parents; i++)
    {
        self->parents[i]->grad = sref(self->grad);
        backward(self->parents[i]);
    }
}

void backward_copy(tensor_t *self)
{
    ASSERT(self->n_parents == 1, "backward_copy expects 1 parent, got %d", self->n_parents);
    init_grad(self->parents[0]);
    iterator_t it = tensor_iterator(self);
    copy_to_range(self->parents[0]->grad, self->grad, &it);
    iterator_free(&it);
    backward(self->parents[0]);
}

void backward_slice(tensor_t *self)
{
    ASSERT(self->n_parents == 1, "backward_slice expects 1 parent, got %d", self->n_parents);
    init_grad(self->parents[0]);
    iterator_t it = tensor_iterator(self);
    copy_to_range(self->parents[0]->grad, self->grad, &it);
    iterator_free(&it);
    sfree(self->grad);
    self->grad = sref(self->parents[0]->grad);
    backward(self->parents[0]);
}
