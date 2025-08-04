#include "ops.h"

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
    self->data = smalloc (.size = self->size, .nmemb = sizeof (float), .kind = SHARED);
    relut (self, self->children[0]);
}

// BINARY OPS
void
forward_add (tensor_t *self)
{
    ASSERT (self->n_children == 2, "forward_add must have 2 children, got %d", self->n_children);
    self->data = smalloc (.size = self->size, .nmemb = sizeof (float), .kind = SHARED);
    addt (self, self->children[0], self->children[1]);
}

void
forward_mul (tensor_t *self)
{
    ASSERT (self->n_children == 2, "forward_mul must have 2 children, got %d", self->n_children);
    self->data = smalloc (.size = self->size, .nmemb = sizeof (float), .kind = SHARED);
    mult (self, self->children[0], self->children[1]);
}

void
forward_pow (tensor_t *self)
{
    ASSERT (self->n_children == 2, "forward_pow must have 2 children, got %d", self->n_children);
    self->data = smalloc (.size = self->size, .nmemb = sizeof (float), .kind = SHARED);
    powt (self, self->children[0], self->children[1]);
}

// REDUCE OPS
void
forward_sum (tensor_t *self)
{
    ASSERT (self->n_children == 1, "forward_sum must have 1 child, got %d", self->n_children);
    self->data = smalloc (.size = self->size, .nmemb = sizeof (float), .kind = SHARED);
    sumt (self, self->children[0]);
}

// MOVEMENT OPS

void
forward_ref (tensor_t *self)
{
    ASSERT (self->n_children == 1, "forward_ref must have 1 child, got %d", self->n_children);
    self->data = sref (self->children[0]->data);
}