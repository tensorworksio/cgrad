#include "forward.h"

void
init_data (tensor_t *self)
{
    if (self->data)
        return;
    self->data = smalloc (.nmemb = self->size, .size = sizeof (float), .kind = SHARED);
}

void
forward (tensor_t *self)
{
    if (self->op == NULL || self->op->forward == NULL)
    {
        log_debug ("Node %p has no forward function.", (void *) self);
        return;
    }
    if (self->children == NULL)
    {
        log_debug ("Node %p has no children.", (void *) self);
        return;
    }
    self->op->forward (self);
}

// FORWARD
void
update_data_relu (tensor_t *self, tensor_t *child)
{
    for (int i = 0; i < self->size; ++i)
    {
        self->data[i] = (child->data[i] > 0.0) ? child->data[i] : 0.0;
    }
}

void
update_data_add (tensor_t *self, tensor_t *child, tensor_t *other)
{
    int j, k;
    for (int i = 0; i < self->size; ++i)
    {
        j             = (child->size == 1) ? 0 : i;
        k             = (other->size == 1) ? 0 : i;
        self->data[i] = child->data[j] + other->data[k];
    }
}

void
update_data_mul (tensor_t *self, tensor_t *child, tensor_t *other)
{
    int j, k;
    for (int i = 0; i < self->size; ++i)
    {
        j             = (child->size == 1) ? 0 : i;
        k             = (other->size == 1) ? 0 : i;
        self->data[i] = child->data[j] * other->data[k];
    }
}

void
update_data_pow (tensor_t *self, tensor_t *child, tensor_t *other)
{
    int j, k;
    for (int i = 0; i < self->size; ++i)
    {
        j             = (child->size == 1) ? 0 : i;
        k             = (other->size == 1) ? 0 : i;
        self->data[i] = powf (child->data[j], other->data[k]);
    }
}

void
update_data_sum (tensor_t *self, tensor_t *child)
{
    *self->data = 0.f;
    for (int i = 0; i < child->size; ++i)
    {
        *self->data += child->data[i];
    }
}

void
update_data_slice (tensor_t *self, tensor_t *child, slice_t *range, int ndim)
{
    smart iterator_t *it  = iterator (range, child->stride, ndim);
    int               idx = 0;
    while (iterator_has_next (it))
    {
        self->data[idx++] = child->data[iterator_next (it)];
    }
}

void
update_data_cat (tensor_t *self, tensor_t *child, slice_t *range, int ndim)
{
    smart iterator_t *it  = iterator (range, self->stride, ndim);
    int               idx = 0;
    while (iterator_has_next (it))
    {
        self->data[iterator_next (it)] = child->data[idx++];
    }
}

// UNARY OPS
void
forward_relu (tensor_t *self)
{
    ASSERT (self->n_children == 1, "forward_relu must have 1 child, got %d", self->n_children);
    init_data (self);
    update_data_relu (self, self->children[0]);
}

// BINARY OPS
void
forward_add (tensor_t *self)
{
    ASSERT (self->n_children == 2, "forward_add must have 2 children, got %d", self->n_children);
    init_data (self);
    update_data_add (self, self->children[0], self->children[1]);
}

void
forward_mul (tensor_t *self)
{
    ASSERT (self->n_children == 2, "forward_mul must have 2 children, got %d", self->n_children);
    init_data (self);
    update_data_mul (self, self->children[0], self->children[1]);
}

void
forward_pow (tensor_t *self)
{
    ASSERT (self->n_children == 2, "forward_pow must have 2 children, got %d", self->n_children);
    init_data (self);
    update_data_pow (self, self->children[0], self->children[1]);
}

// REDUCE OPS
void
forward_sum (tensor_t *self)
{
    ASSERT (self->n_children == 1, "forward_sum must have 1 child, got %d", self->n_children);
    init_data (self);
    update_data_sum (self, self->children[0]);
}

// MOVEMENT OPS

void
forward_ref (tensor_t *self)
{
    ASSERT (self->n_children == 1, "forward_ref must have 1 child, got %d", self->n_children);
    self->data = sref (self->children[0]->data);
}

void
forward_copy (tensor_t *self)
{
    ASSERT (self->n_children == 1, "forward_copy must have 1 child, got %d", self->n_children);
    init_data (self);
    memcpy (self->data, self->children[0]->data, self->size * sizeof (float));
}

void
forward_slice (tensor_t *self)
{
    ASSERT (self->n_children == 1, "forward_slice must have 1 child, got %d", self->n_children);

    slice_params_t *params = (slice_params_t *) self->op->params;
    ASSERT (params != NULL && params->range != NULL,
            "Slice parameters must be set for forward_slice");

    init_data (self);
    update_data_slice (self, self->children[0], params->range, params->ndim);
}

void
forward_cat (tensor_t *self)
{
    ASSERT (self->n_children >= 1, "forward_cat must have at least 1 child, got %d",
            self->n_children);

    cat_params_t *params = (cat_params_t *) self->op->params;
    ASSERT (params != NULL, "Cat parameters must be set for forward_cat");

    init_data (self);

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
        stop            = start + child->shape[axis];
        range[axis]     = SLICE_RANGE (start, stop);
        update_data_cat (self, child, range, self->ndim);
        start = stop;
    }
}