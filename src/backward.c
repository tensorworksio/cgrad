#include "backward.h"

void
init_grad (tensor_t *self)
{
    if (!self->requires_grad || self->grad)
        return;
    self->grad = smalloc (.nmemb = self->size, .size = sizeof (float), .kind = SHARED);
    tensor_zero_grad (self);
}

void
backward (tensor_t *self)
{
    if (self->op == NULL)
    {
        return;
    }

    self->op->backward (self);
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

void
update_grad_sum_dim (tensor_t *self, tensor_t *child)
{
    if (!child->requires_grad)
        return;

    axis_params_t *params = (axis_params_t *) self->op->params;
    int            axis   = params->axis;
    int            dim    = child->shape[axis];
    int            stride = child->stride[axis];

    slice_t range[self->ndim];
    for (int i = 0; i < self->ndim; ++i)
    {
        range[i] = SLICE_RANGE (0, self->shape[i]);
    }

    smart iterator_t *it = iterator (range, self->stride, self->ndim);

    while (iterator_has_next (it))
    {
        int input_offset = 0;
        for (int d = 0; d < self->ndim; ++d)
        {
            input_offset += it->indices[d] * child->stride[d];
        }

        int   out_idx = iterator_next (it);
        float grad    = self->grad[out_idx];

        // Broadcast gradient along the axis
        for (int k = 0; k < dim; ++k)
        {
            child->grad[input_offset + k * stride] += grad;
        }
    }
}

void
update_grad_slice (tensor_t *self, tensor_t *child, slice_t *range, int ndim)
{
    if (!child->requires_grad)
        return;

    smart iterator_t *it  = iterator (range, child->stride, ndim);
    int               idx = 0;
    while (iterator_has_next (it))
    {
        int pos = iterator_next (it);
        child->grad[pos] += self->grad[idx++];
    }
}

void
update_grad_cat (tensor_t *self, tensor_t *child, slice_t *range, int ndim)
{
    if (!child->requires_grad)
        return;

    smart iterator_t *it  = iterator (range, self->stride, ndim);
    int               idx = 0;
    while (iterator_has_next (it))
    {
        child->grad[idx++] += self->grad[iterator_next (it)];
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
}

// REDUCE OPS
void
backward_sum (tensor_t *self)
{
    ASSERT (self->n_children == 1, "SUM Node %p expects 1 child, got %d", (void *) self,
            self->n_children);
    init_grad (self->children[0]);

    if (self->op->params == NULL)
    {
        update_grad_sum (self, self->children[0]);
    }
    else
    {
        update_grad_sum_dim (self, self->children[0]);
    }
}

// MOVEMENT OPS
void
backward_ref (tensor_t *self)
{
    ASSERT (self->n_children == 1, "backward_ref expects 1 child, got %d", self->n_children);
    self->children[0]->grad = sref (self->grad);
}

void
backward_copy (tensor_t *self)
{
    ASSERT (self->n_children == 1, "backward_copy expects 1 child, got %d", self->n_children);
    init_grad (self->children[0]);

    tensor_t *child = self->children[0];
    if (!child->requires_grad)
        return;

    slice_t range[child->ndim];
    for (int i = 0; i < child->ndim; ++i)
    {
        range[i] = SLICE_RANGE (0, child->shape[i]);
    }

    smart iterator_t *it  = iterator (range, child->stride, child->ndim);
    int               idx = 0;
    while (iterator_has_next (it))
    {
        child->grad[iterator_next (it)] += self->grad[idx++];
    }
}

void
backward_slice (tensor_t *self)
{
    ASSERT (self->n_children == 1, "backward_slice expects 1 child, got %d", self->n_children);

    slice_params_t *params = (slice_params_t *) self->op->params;
    ASSERT (params != NULL && params->range != NULL,
            "Slice parameters must be set for backward_slice");

    init_grad (self->children[0]);
    update_grad_slice (self, self->children[0], params->range, params->ndim);
}

void
backward_cat (tensor_t *self)
{
    ASSERT (self->n_children >= 1, "backward_cat must have at least 1 child, got %d",
            self->n_children);

    axis_params_t *params = (axis_params_t *) self->op->params;
    ASSERT (params != NULL, "Cat parameters must be set for backward_cat");

    for (int i = 0; i < self->n_children; i++)
        init_grad (self->children[i]);

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
        stop                 = start + child->shape[axis];
        range[axis]          = SLICE_RANGE (start, stop);
        smart iterator_t *it = iterator (range, self->stride, self->ndim);
        update_grad_cat (self, child, range, self->ndim);
        start = stop;
    }
}