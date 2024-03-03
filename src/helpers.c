#include "helpers.h"

int get_size(int shape[], int ndim)
{
    int size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size *= shape[i];
    }
    return size;
}

void swap_int(int *array, int index1, int index2)
{
    int temp = array[index1];
    array[index1] = array[index2];
    array[index2] = temp;
}

void swap_slice(slice_t *array, int index1, int index2)
{
    slice_t temp = array[index1];
    array[index1] = array[index2];
    array[index2] = temp;
}

void set_data(float *data, float value, int size)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = value;
    }
}

bool is_same_shape(int shape_a[], int shape_b[], int ndim_a, int ndim_b)
{
    if (ndim_a != ndim_b)
        return false;

    for (int i = 0; i < ndim_a; i++)
    {
        if (shape_a[i] != shape_b[i])
            return false;
    }
    return true;
}

bool is_equal_data(float *data_a, float *data_b, int size)
{
    if (data_a == NULL || data_b == NULL)
        return false;
    for (int i = 0; i < size; i++)
    {
        if (fabs(data_a[i] - data_b[i]) > EPSILON)
            return false;
    }
    return true;
}

void normalize_range(slice_t range[], int shape[], int ndim)
{
    for (int d = 0; d < ndim; d++)
    {
        ASSERT(range[d].step > 0, "Slice step must be positive: got %d", range[d].step);
        ASSERT(range[d].start < shape[d] && range[d].start >= -shape[d],
               "Index %d is out of bounds for dimension %d with size %d", range[d].start, d, shape[d]);
        ASSERT(range[d].stop <= shape[d] && range[d].stop > -shape[d],
               "Index %d is out of bounds for dimension %d with size %d", range[d].stop, d, shape[d]);

        // if start/stop is negative, add shape to it
        range[d].start = (range[d].start < 0) ? range[d].start + shape[d] : range[d].start;
        range[d].stop = (range[d].stop < 0) ? range[d].stop + shape[d] + 1 : range[d].stop;

        ASSERT(range[d].start <= range[d].stop, "Slice start must be less than or equal to slice stop");
    }
}

void copy_from_range(float *dst, float *src, iterator_t *it)
{
    int idx = 0;
    while (iterator_has_next(it))
    {
        dst[idx++] = src[iterator_next(it)];
    }
}

void copy_to_range(float *dst, float *src, iterator_t *it)
{
    int idx = 0;
    while (iterator_has_next(it))
    {
        dst[iterator_next(it)] = src[idx++];
    }
}

void print_metadata(int *data, int ndim)
{
    printf("[");
    for (int i = 0; i < ndim - 1; i++)
    {
        printf("%d, ", data[i]);
    }
    printf("%d]", data[ndim - 1]);
}

void print_data(float *data, iterator_t *it)
{
    int eod;
    int index;
    while (iterator_has_next(it))
    {
        eod = iterator_eod(it);
        index = iterator_next(it);
        printf("%f ", data[index]);
        for (int i = 0; i < eod; i++)
            printf("\n");
    }
}