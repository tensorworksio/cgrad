#include "iterator.h"

void
iterator_destructor (void *ptr, void *meta)
{
    iterator_t *it = (iterator_t *) ptr;
    free (it->shape);
    free (it->indices);
    free (it->range);
    free (it->stride);
}

iterator_t *
iterator (slice_t *range, int *stride, int ndim)
{
    iterator_t *it = unique_ptr (
        iterator_t, { .ndim = ndim, .range = NULL, .shape = NULL, .stride = NULL, .indices = NULL },
        iterator_destructor);

    it->shape   = (int *) malloc (ndim * sizeof (int));
    it->indices = (int *) malloc (ndim * sizeof (int));
    it->range   = (slice_t *) malloc (ndim * sizeof (slice_t));
    memcpy (it->range, range, ndim * sizeof (slice_t));
    it->stride = (int *) malloc (ndim * sizeof (int));
    memcpy (it->stride, stride, ndim * sizeof (int));

    iterator_reset (it);
    return it;
}

void
iterator_reset (iterator_t *it)
{
    memset (it->indices, 0, it->ndim * sizeof (int));
    for (int i = 0; i < it->ndim; i++)
    {
        it->shape[i] = SLICE_SIZE (it->range[i]);
    }
    it->has_next = (iterator_size (it) > 0);
}

void
iterator_free (iterator_t *it)
{
    sfree (it);
}

int
iterator_size (iterator_t *it)
{
    int size = 1;
    for (int i = 0; i < it->ndim; i++)
    {
        size *= it->shape[i];
    }
    return size;
}

int
iterator_next (iterator_t *it)
{
    ASSERT (iterator_has_next (it), "No more elements to iterate over");
    // Inline index calculation
    int index = 0;
    for (int i = 0; i < it->ndim; i++)
    {
        index += (it->range[i].start + it->indices[i] * it->range[i].step) * it->stride[i];
    }
    // Inline update logic
    for (int i = it->ndim; i-- > 0;)
    {
        if (it->indices[i] < it->shape[i] - 1)
        {
            it->indices[i] += 1;
            break;
        }
        else
        {
            it->indices[i] = 0;
            if (i == 0)
            {
                it->has_next = false;
                break;
            }
        }
    }
    return index;
}

bool
iterator_has_next (iterator_t *it)
{
    return it->has_next;
}