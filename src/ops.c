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

// TODO:
// before any operation, check if full range is used
// if so, iterate over data contiguous memory
// if not, use iterator to iterate over data

// FORWARD
void relut(tensor_t *self, tensor_t *parent)
{
    for (int i = 0; i < self->size; i++)
    {
        self->data[i] = (parent->data[i] > 0.0) ? parent->data[i] : 0.0;
    }
}

void addt(tensor_t *self, tensor_t *parent, tensor_t *other)
{
    int j, k;
    for (int i = 0; i < self->size; i++)
    {
        j = (parent->size == 1) ? 0 : i;
        k = (other->size == 1) ? 0 : i;
        self->data[i] = parent->data[j] + other->data[k];
    }
}

void mult(tensor_t *self, tensor_t *parent, tensor_t *other)
{
    int j, k;
    for (int i = 0; i < self->size; i++)
    {
        j = (parent->size == 1) ? 0 : i;
        k = (other->size == 1) ? 0 : i;
        self->data[i] = parent->data[j] * other->data[k];
    }
}

void powt(tensor_t *self, tensor_t *parent, tensor_t *other)
{
    int j, k;
    for (int i = 0; i < self->size; i++)
    {
        j = (parent->size == 1) ? 0 : i;
        k = (other->size == 1) ? 0 : i;
        self->data[i] = powf(parent->data[j], other->data[k]);
    }
}

void sumt(tensor_t *self, tensor_t *parent)
{
    *self->data = 0.0;
    iterator_t it = tensor_iterator(parent);
    while (iterator_has_next(&it))
    {
        *self->data += parent->data[iterator_next(&it)];
    }
    iterator_free(&it);
}

void catt(tensor_t *self, tensor_t *parents[], int n_parents)
{
    for (int i = 0; i < n_parents; i++)
    {
        iterator_t it = tensor_iterator(parents[i]);
        copy_to_range(self->data, parents[i]->data, &it);
        iterator_free(&it);
        sfree(parents[i]->data);
        parents[i]->data = sref(self->data);
    }
}

// UNARY OPS
void forward_relu(tensor_t *self)
{
    ASSERT(self->n_parents == 1, "forward_relu must have 1 parent, got %d", self->n_parents);
    init_data(self);
    relut(self, self->parents[0]);
}

// BINARY OPS
void forward_add(tensor_t *self)
{
    ASSERT(self->n_parents == 2, "forward_add must have 2 parents, got %d", self->n_parents);
    init_data(self);
    addt(self, self->parents[0], self->parents[1]);
}

void forward_mul(tensor_t *self)
{
    ASSERT(self->n_parents == 2, "forward_mul must have 2 parents, got %d", self->n_parents);
    init_data(self);
    mult(self, self->parents[0], self->parents[1]);
}

void forward_pow(tensor_t *self)
{
    ASSERT(self->n_parents == 2, "forward_pow must have 2 parents, got %d", self->n_parents);
    init_data(self);
    powt(self, self->parents[0], self->parents[1]);
}

// REDUCE OPS
void forward_sum(tensor_t *self)
{
    ASSERT(self->n_parents == 1, "forward_sum must have 1 parent, got %d", self->n_parents);
    init_data(self);
    sumt(self, self->parents[0]);
}

// MOVEMENT OPS
void forward_cat(tensor_t *self)
{
    ASSERT(self->n_parents > 0, "forward_cat must have at least 1 parent, got %d", self->n_parents);
    init_data(self);
    catt(self, self->parents, self->n_parents);
}

void forward_copy(tensor_t *self)
{
    ASSERT(self->n_parents == 1, "forward_copy must have 1 parent, got %d", self->n_parents);
    init_data(self);
    iterator_t it = tensor_iterator(self);
    copy_from_range(self->data, self->parents[0]->data, &it);
    iterator_free(&it);
}

void forward_ref(tensor_t *self)
{
    ASSERT(self->n_parents == 1, "forward_ref must have 1 parent, got %d", self->n_parents);
    self->data = sref(self->parents[0]->data);
}
