#include "iterator.h"

iterator_t iterator(slice_t *range, int *stride, int ndim)
{
    iterator_t it;
    it.ndim = ndim;
    it.range = range;
    it.stride = stride;

    it.shape = (int *)malloc(ndim * sizeof(int));
    it.indices = (int *)malloc(ndim * sizeof(int));

    for (int i = 0; i < ndim; i++)
    {
        it.indices[i] = 0;
        it.shape[i] = slice_size(range[i]);
    }

    it.has_next = (iterator_size(&it) > 0) ? true : false;

    return it;
}

void iterator_free(iterator_t *it)
{
    free(it->shape);
    free(it->indices);
    it->shape = NULL;
    it->indices = NULL;
}

int iterator_size(iterator_t *it)
{
    int size = 1;
    for (int i = 0; i < it->ndim; i++)
    {
        size *= it->shape[i];
    }
    return size;
}

int iterator_eod(iterator_t *it)
{
    int eod = 0;
    for (int i = it->ndim; i-- > 0;)
    {
        if (it->indices[i] != it->shape[i] - 1)
        {
            return eod;
        }
        eod++;
    }
    return eod;
}

int iterator_sod(iterator_t *it)
{
    int sod = 0;
    for (int i = it->ndim; i-- > 0;)
    {
        if (it->indices[i] != 0)
        {
            return sod;
        }
        sod++;
    }
    return sod;
}

void iterator_update(iterator_t *it)
{
    for (int i = it->ndim; i-- > 0;)
    {
        if (it->indices[i] < it->shape[i] - 1)
        {
            it->indices[i] += 1;
            break;
        }
        else
        {
            // Reset the current index and carry over to the next dimension
            it->indices[i] = 0;
            if (i == 0)
            {
                it->has_next = false;
                break;
            }
        }
    }
}

bool iterator_has_next(iterator_t *it)
{
    return it->has_next;
}

int iterator_index(iterator_t *it)
{
    int index = 0;
    for (int i = 0; i < it->ndim; i++)
    {
        index += (it->range[i].start + it->indices[i] * it->range[i].step) * it->stride[i];
    }
    return index;
}

int iterator_next(iterator_t *it)
{
    ASSERT(iterator_has_next(it), "No more elements to iterate over");
    // Get index for the current iteration
    int index = iterator_index(it);
    // Update indices for the next iteration
    iterator_update(it);

    return index;
}