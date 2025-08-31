#include "ops.h"

void
init_data (tensor_t *self)
{
    if (self->data)
        return;
    self->data = smalloc (.size = self->size, .nmemb = sizeof (float), .kind = SHARED);
}

// FORWARD
void
relut (tensor_t *self, tensor_t *child)
{
    for (int i = 0; i < self->size; ++i)
    {
        self->data[i] = (child->data[i] > 0.0) ? child->data[i] : 0.0;
    }
}

void
addt (tensor_t *self, tensor_t *child, tensor_t *other)
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
mult (tensor_t *self, tensor_t *child, tensor_t *other)
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
powt (tensor_t *self, tensor_t *child, tensor_t *other)
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
sumt (tensor_t *self, tensor_t *child)
{
    *self->data = 0.f;
    for (int i = 0; i < child->size; ++i)
    {
        *self->data += child->data[i];
    }
}

// UNARY OPS
void
forward_relu (tensor_t *self)
{
    ASSERT (self->n_children == 1, "forward_relu must have 1 child, got %d", self->n_children);
    init_data (self);
    relut (self, self->children[0]);
}

// BINARY OPS
void
forward_add (tensor_t *self)
{
    ASSERT (self->n_children == 2, "forward_add must have 2 children, got %d", self->n_children);
    init_data (self);
    addt (self, self->children[0], self->children[1]);
}

void
forward_mul (tensor_t *self)
{
    ASSERT (self->n_children == 2, "forward_mul must have 2 children, got %d", self->n_children);
    init_data (self);
    mult (self, self->children[0], self->children[1]);
}

void
forward_pow (tensor_t *self)
{
    ASSERT (self->n_children == 2, "forward_pow must have 2 children, got %d", self->n_children);
    init_data (self);
    powt (self, self->children[0], self->children[1]);
}

// REDUCE OPS
void
forward_sum (tensor_t *self)
{
    ASSERT (self->n_children == 1, "forward_sum must have 1 child, got %d", self->n_children);
    init_data (self);
    sumt (self, self->children[0]);
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
    ASSERT (self->range != NULL, "Slice range must be set for forward_slice");

    init_data (self);

    iterator_t it = iterator (self->range, self->children[0]->stride, self->ndim);
    copy_from_range (self->data, self->children[0]->data, &it);
    iterator_free (&it);
}