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

int get_index(int coords[], int shape[], int ndim)
{
    int index = 0;
    int multiplier = 1;
    for (int i = ndim - 1; i >= 0; i--)
    {
        index += coords[i] * multiplier;
        multiplier *= shape[i];
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

void normalize_ranges(slice_t ranges[], int shape[], int ndim)
{   
    for (int d = 0; d < ndim; d++)
    {   
        ASSERT(ranges[d].step > 0, "Slice step must be positive: got %d", ranges[d].step);
        ASSERT(-shape[d] <= ranges[d].start && ranges[d].start < shape[d],
               "Index %d is out of bounds for dimension %d with size %d", ranges[d].start, d, shape[d]);
        ASSERT(-shape[d] < ranges[d].stop && ranges[d].stop <= shape[d],
               "Index %d is out of bounds for dimension %d with size %d", ranges[d].stop, d, shape[d]);

        // if start/stop is negative, add shape to it
        if (ranges[d].start < 0)
            ranges[d].start += (shape[d] + 1);
        if (ranges[d].stop < 0)
            ranges[d].stop += (shape[d] + 1);

        ASSERT(ranges[d].start <= ranges[d].stop, "Slice start must be less than or equal to slice stop");
    }
}

void compute_shape(int shape[], slice_t ranges[], int ndim)
{   
    // Compute shape from ranges
    int range;
    for (int d = 0; d < ndim; d++)
    {
        range = abs(ranges[d].stop - ranges[d].start);
        shape[d] = range / ranges[d].step + (range % ranges[d].step != 0 ? 1 : 0);
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

void print_data_ndim(float *data, int shape[], int stride[], int indices[], int ndim, int dim)
{
    if (dim == ndim) {
        int index = 0;
        for (int i = 0; i < ndim; i++) {
            index += stride[i] * indices[i];
        }
        printf("%f ", data[index]);
        return;
    }

    for (indices[dim] = 0; indices[dim] < shape[dim]; indices[dim]++) {
        print_data_ndim(data, shape, stride, indices, ndim, dim + 1);
    }
    printf("\n");
}

void print_data(float *data, int shape[], int stride[], int ndim)
{
    int indices[ndim];
    for (int i = 0; i < ndim; i++) indices[i] = 0;
    print_data_ndim(data, shape, stride, indices, ndim, 0);
}