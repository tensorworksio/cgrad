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

iterator_t iterator_tensor(tensor_t *tensor)
{
    return iterator(tensor->range, tensor->stride, tensor->ndim);
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

void iterator_free(iterator_t *it)
{
    free(it->indices);
    it->indices = NULL;
}

bool iterator_has_next(iterator_t *it)
{
    return it->count < it->size;
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

int iterator_next(iterator_t *it)
{
    ASSERT(iterator_has_next(it), "No more elements to iterate over.");
    // TODO: if canonical stride, just return count as index

    int index = get_index(it->indices, it->stride, it->ndim);
    it->count++;

    // Update indices for the next iteration
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

    return index;
}