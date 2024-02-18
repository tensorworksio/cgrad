#include "ops.h"

void init_data(tensor_t *self)
{
    if (self->data)
        return;
    self->data = smalloc(.size = self->size, .nmemb = sizeof(float), .kind = SHARED);
}

void free_data(tensor_t *self)
{
    if (self->data)
    {
        sfree(self->data);
        self->data = NULL;
    }
}

void copy_data(tensor_t *dst, tensor_t *src)
{
    copy_forward(dst->data, src->data, dst->range, src->stride, src->ndim);
}

// FORWARD
void forward_relu(tensor_t *self)
{
    ASSERT(self->n_children == 1, "Relu forward must have 1 child, got %d", self->n_children);
    init_data(self);
    relut(self, self->children[0]);
}

void forward_add(tensor_t *self)
{
    ASSERT(self->n_children == 2, "Add forward must have 2 children, got %d", self->n_children);
    init_data(self);
    addt(self, self->children[0], self->children[1]);
}

void forward_mul(tensor_t *self)
{
    ASSERT(self->n_children == 2, "Mul forward must have 2 children, got %d", self->n_children);
    init_data(self);
    mult(self, self->children[0], self->children[1]);
}

void forward_pow(tensor_t *self)
{
    ASSERT(self->n_children == 2, "Pow forward must have 2 children, got %d", self->n_children);
    init_data(self);
    powt(self, self->children[0], self->children[1]);
}

void forward_sum(tensor_t *self)
{
    ASSERT(self->n_children == 1, "Sum forward must have 1 child, got %d", self->n_children);
    init_data(self);
    sumt(self, self->children[0]);
}

void forward_slice(tensor_t *self)
{
    ASSERT(self->n_children == 1, "Slice forward must have 1 child, got %d", self->n_children);
    init_data(self);
    copy_data(self, self->children[0]);
}

void forward_cat(tensor_t *self)
{
    ASSERT(self->n_children > 1, "Cat forward must have more than 1 child, got %d", self->n_children);
    init_data(self);
    catt(self, self->children, self->n_children);
}

void forward_noop(tensor_t *self)
{
    ASSERT(self->n_children == 1, "Noop forward must have 1 child, got %d", self->n_children);
    self->data = sref(self->children[0]->data);
}

// TODO:
// before any operation, check if full range is used
// if so, iterate over data contiguous memory
// if not, use iterator to iterate over data

// UNARY OPS
void relut(tensor_t *self, tensor_t *child)
{
    for (int i = 0; i < self->size; i++)
    {
        self->data[i] = (child->data[i] > 0.0) ? child->data[i] : 0.0;
    }
}

// BINARY OPS
void addt(tensor_t *self, tensor_t *child, tensor_t *other)
{
    int j, k;
    for (int i = 0; i < self->size; i++)
    {
        j = (child->size == 1) ? 0 : i;
        k = (other->size == 1) ? 0 : i;
        self->data[i] = child->data[j] + other->data[k];
    }
}

void mult(tensor_t *self, tensor_t *child, tensor_t *other)
{
    int j, k;
    for (int i = 0; i < self->size; i++)
    {
        j = (child->size == 1) ? 0 : i;
        k = (other->size == 1) ? 0 : i;
        self->data[i] = child->data[j] * other->data[k];
    }
}

void powt(tensor_t *self, tensor_t *child, tensor_t *other)
{
    int j, k;
    for (int i = 0; i < self->size; i++)
    {
        j = (child->size == 1) ? 0 : i;
        k = (other->size == 1) ? 0 : i;
        self->data[i] = powf(child->data[j], other->data[k]);
    }
}

// REDUCE OPS
void sumt(tensor_t *self, tensor_t *child)
{
    for (int i = 0; i < child->size; i++)
    {
        self->data[0] += child->data[i];
    }
}

// MOVEMENT OPS
void catt(tensor_t *self, tensor_t *children[], int n_children)
{
    int axis;
    int step;
    int size;
    int n;
    int offset = 0;

    for (int d = 0; d < self->ndim; d++)
    {
        if (self->shape[d] > children[0]->shape[d])
        {
            axis = d;
            break;
        }
    }

    for (int i = 0; i < n_children; i++)
    {
        step = 0;
        size = (axis == 0) ? children[i]->size : children[i]->stride[axis - 1];
        n = children[i]->size / size;
        for (int j = 0; j < n; j++)
        {
            memcpy(self->data + offset + step, children[i]->data + j * size, size * sizeof(float));
            step += self->stride[axis - 1];
        }
        offset += size;

        // assign child to self
        // free_data(children[i]);
        // children[i]->data = sref(self->data);
        // memcpy(children[i]->stride, self->stride, self->ndim * sizeof(int));
        // memcpy(children[i]->range, self->range, self->ndim * sizeof(slice_t));
        // TODO set range[axis] to the correct value
    }
}