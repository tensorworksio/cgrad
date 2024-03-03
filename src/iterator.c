#include "iterator.h"

iterator_t iterator(slice_t *range, int *stride, int ndim)
{
    iterator_t it;
    it.ndim = ndim;
    it.range = range;
    it.stride = stride;
    it.indices = (int *)malloc(ndim * sizeof(int));
    it.size = iterator_size(&it);
    iterator_reset(&it);

    return it;
}

void iterator_free(iterator_t *it)
{
    free(it->indices);
    it->indices = NULL;
}

void iterator_reset(iterator_t *it)
{
    it->count = 0;
    for (int i = 0; i < it->ndim; i++)
    {
        it->indices[i] = it->range[i].start;
    }
}

int iterator_size(iterator_t *it)
{
    int size = 1;
    for (int i = 0; i < it->ndim; i++)
    {
        size *= (it->range[i].stop - it->range[i].start) / it->range[i].step;
    }
    return size;
}

int iterator_eod(iterator_t *it)
{
    int eod = 0;
    for (int i = it->ndim; i-- > 0;)
    {
        if (it->indices[i] != it->range[i].stop - it->range[i].step)
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
        if (it->indices[i] != it->range[i].start)
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
        if (it->indices[i] + it->range[i].step < it->range[i].stop)
        {
            it->indices[i] += it->range[i].step;
            break;
        }
        else if (i != 0)
        {
            // Reset the current index and carry over to the next dimension
            it->indices[i] = it->range[i].start;
        }
    }
    it->count++;
}

bool iterator_has_next(iterator_t *it)
{
    return it->count < it->size;
}

int iterator_index(iterator_t *it)
{
    int index = 0;
    for (int i = 0; i < it->ndim; i++)
    {
        index += it->indices[i] * it->stride[i];
    }
    return index;
}

int iterator_next(iterator_t *it)
{
    ASSERT(iterator_has_next(it), "No more elements to iterate over.");
    // Get index for the current iteration
    int index = iterator_index(it);
    // Update indices for the next iteration
    iterator_update(it);

    return index;
}