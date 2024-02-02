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

int get_index(int coords[], int stride[], int ndim)
{
    int index = 0;
    for (int i = 0; i < ndim; i++)
    {
        index += coords[i] * stride[i];
    }
    return index;
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
        ASSERT(abs(range[d].start) <= shape[d],
               "Index %d is out of bounds for dimension %d with size %d", range[d].start, d, shape[d]);
        ASSERT(abs(range[d].stop) <= shape[d],
               "Index %d is out of bounds for dimension %d with size %d", range[d].stop, d, shape[d]);

        // if start/stop is negative, add shape to it
        range[d].start = (range[d].start < 0) ? range[d].start + shape[d] : range[d].start;
        range[d].stop = (range[d].stop < 0) ? range[d].stop + shape[d] : range[d].stop;

        ASSERT(range[d].start <= range[d].stop, "Slice start must be less than or equal to slice stop");
    }
}

void compute_shape(int shape[], slice_t range[], int ndim)
{   
    int dist;
    for (int d = 0; d < ndim; d++)
    {
        dist = abs(range[d].stop - range[d].start);
        shape[d] = dist / range[d].step + (dist % range[d].step != 0 ? 1 : 0);
    }
}

void print_metadata(int data[], int ndim)
{   
    printf("[");
    for (int i = 0; i < ndim-1; i++)
    {
        printf("%d, ", data[i]);
    }
    printf("%d]", data[ndim-1]);
}

void print_data_ndim(float *data, slice_t range[], int stride[], int indices[], int ndim, int dim)
{
    if (dim == ndim) {
        int index = get_index(indices, stride, ndim);
        printf("%f ", data[index]);
        return;
    }

    for (indices[dim] = range[dim].start; indices[dim] < range[dim].stop; indices[dim] += range[dim].step) {
        print_data_ndim(data, range, stride, indices, ndim, dim + 1);
    }
    printf("\n");
}

void print_data(float *data, slice_t range[], int stride[], int ndim)
{
    int indices[ndim];
    for (int i = 0; i < ndim; i++) indices[i] = 0;
    print_data_ndim(data, range, stride, indices, ndim, 0);
}