#include "backward.h"

void
backward (tensor_t *self)
{
    if (self->op == NULL || self->op->backward == NULL)
    {
        log_debug ("Node %p has no backward function.", (void *) self);
        return;
    }
    if (self->children == NULL)
    {
        log_debug ("Node %p has no children.", (void *) self);
        return;
    }
    self->op->backward (self);
}

void
init_grad (tensor_t *self)
{
    if (!self->requires_grad || self->grad)
        return;
    self->grad = smalloc (.nmemb = self->size, .size = sizeof (float), .kind = SHARED);
    tensor_zero_grad (self);
}

// UPDATE OPS
void
update_grad_add (tensor_t *self, tensor_t *child)
{
    if (!child->requires_grad)
        return;

    for (int i = 0; i < self->size; i++)
    {
        child->grad[i] += self->grad[i];
    }
}

void
update_grad_relu (tensor_t *self, tensor_t *child)
{
    if (!child->requires_grad)
        return;
    for (int i = 0; i < self->size; i++)
    {
        child->grad[i] += self->grad[i] * (self->data[i] > 0);
    }
}

void
update_grad_mul (tensor_t *self, tensor_t *child, tensor_t *other)
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

void
update_grad_pow (tensor_t *self, tensor_t *child, tensor_t *other)
{
    if (!child->requires_grad)
        return;

    int k;
    for (int i = 0; i < self->size; i++)
    {
        k = (other->size == 1) ? 0 : i;
        child->grad[i]
            += self->grad[i] * other->data[k] * powf (child->data[i], other->data[k] - 1);
    }
}

void
update_grad_exp (tensor_t *self, tensor_t *child, tensor_t *other)
{
    if (!child->requires_grad)
        return;

    int k;
    for (int i = 0; i < self->size; i++)
    {
        k = (other->size == 1) ? 0 : i;
        child->grad[i] += self->grad[i] * self->data[i] * logf (other->data[k]);
    }
}

void
update_grad_sum (tensor_t *self, tensor_t *child)
{
    if (!child->requires_grad)
        return;
    for (int i = 0; i < child->size; i++)
    {
        child->grad[i] += self->grad[0];
    }
}

// UNARY OPS
void
backward_relu (tensor_t *self)
{
    ASSERT (self->n_children == 1, "RELU Node %p expects 1 child, got %d", (void *) self,
            self->n_children);
    init_grad (self->children[0]);
    update_grad_relu (self, self->children[0]);
    backward (self->children[0]);
}

// BINARY OPS
void
backward_add (tensor_t *self)
{
    ASSERT (self->n_children == 2, "ADD Node %p expects 2 children, got %d", (void *) self,
            self->n_children);
    init_grad (self->children[0]);
    init_grad (self->children[1]);

    update_grad_add (self, self->children[0]);
    update_grad_add (self, self->children[1]);

    backward (self->children[0]);
    backward (self->children[1]);
}

void
backward_mul (tensor_t *self)
{
    ASSERT (self->n_children == 2, "MUL Node %p expects 2 children, got %d", (void *) self,
            self->n_children);
    init_grad (self->children[0]);
    init_grad (self->children[1]);

    update_grad_mul (self, self->children[0], self->children[1]);
    update_grad_mul (self, self->children[1], self->children[0]);

    backward (self->children[0]);
    backward (self->children[1]);
}

void
backward_pow (tensor_t *self)
{
    ASSERT (self->n_children == 2, "POW Node %p expects 2 children, got %d", (void *) self,
            self->n_children);
    init_grad (self->children[0]);
    init_grad (self->children[1]);

    update_grad_pow (self, self->children[0], self->children[1]);
    update_grad_exp (self, self->children[1], self->children[0]);

    backward (self->children[0]);
    backward (self->children[1]);
}

// REDUCE OPS
void
backward_sum (tensor_t *self)
{
    ASSERT (self->n_children == 1, "SUM Node %p expects 1 child, got %d", (void *) self,
            self->n_children);
    init_grad (self->children[0]);
    update_grad_sum (self, self->children[0]);
    backward (self->children[0]);
}

// MOVEMENT OPS
void
backward_ref (tensor_t *self)
{
    ASSERT (self->n_children == 1, "backward_ref expects 1 child, got %d", self->n_children);
    self->children[0]->grad = sref (self->grad);
    backward (self->children[0]);
}

void
backward_copy (tensor_t *self)
{
    ASSERT (self->n_children == 1, "backward_copy expects 1 child, got %d", self->n_children);
    init_grad (self->children[0]);
    update_grad_add (self, self->children[0]);
    backward (self->children[0]);
}

void
backward_slice (tensor_t *self)
{
    ASSERT (self->n_children == 1, "backward_slice expects 1 child, got %d", self->n_children);

    slice_params_t *params = (slice_params_t *) self->op->params;
    ASSERT (params != NULL && params->range != NULL,
            "Slice parameters must be set for backward_slice");

    init_grad (self->children[0]);

    smart iterator_t *it = iterator (params->range, self->children[0]->stride, self->ndim);
    copy_to_range (self->children[0]->grad, self->grad, it);
    backward (self->children[0]);
}

void
backward_cat (tensor_t *self)
{
    ASSERT (self->n_children >= 1, "backward_cat must have at least 1 child, got %d",
            self->n_children);

    cat_params_t *params = (cat_params_t *) self->op->params;
    ASSERT (params != NULL, "Cat parameters must be set for backward_cat");
    int axis = params->axis;

    slice_t range[self->ndim];
    for (int d = 0; d < self->ndim; d++)
    {
        range[d] = SLICE_RANGE (0, self->shape[d]);
    }

    int start = 0, stop = 0;
    for (int i = 0; i < self->n_children; i++)
    {
        tensor_t *child = self->children[i];
        if (!child->requires_grad)
        {
            start += child->shape[axis];
            continue;
        }
        init_grad (child);
        stop                 = start + child->shape[axis];
        range[axis]          = SLICE_RANGE (start, stop);
        smart iterator_t *it = iterator (range, self->stride, self->ndim);
        copy_from_range (child->grad, self->grad, it);
        start = stop;
        backward (child);
    }
}