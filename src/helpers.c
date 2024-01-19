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

int get_index(int shape[], int coords[], int ndim)
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

void set_ranges(slice_t ranges[], int shape[], int ndim)
{
    for (int d = 0; d < ndim; d++)
    {
        // if start/stop is negative, add shape to it
        while (ranges[d].start < 0)
            ranges[d].start += shape[d];
        while (ranges[d].stop < 0)
            ranges[d].stop += shape[d];
        // if start/stop is greater than shape, set it to shape
        if (ranges[d].start > shape[d])
            ranges[d].start = shape[d];
        if (ranges[d].stop > shape[d])
            ranges[d].stop = shape[d];
    }
}

void set_shape(int shape[], slice_t ranges[], int ndim)
{
    int range;
    for (int d = 0; d < ndim; d++)
    {
        range = abs(ranges[d].stop - ranges[d].start);
        shape[d] = range / ranges[d].step + (range % ranges[d].step != 0 ? 1 : 0);
    }
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

void print_data_ndim(float *data, int shape[], int stride[], int indices[], int ndim, int current_dim)
{
    if (current_dim == ndim)
    {
        // Compute the index in the flat data array
        int index = 0;
        for (int i = 0; i < ndim; i++)
        {
            index += stride[i] * indices[i];
        }

        // Print the data at this index
        printf("%f ", data[index]);
    }
    else
    {
        // Iterate over the current dimension
        for (indices[current_dim] = 0; indices[current_dim] < shape[current_dim]; indices[current_dim]++)
        {
            print_data_ndim(data, shape, stride, indices, ndim, current_dim + 1);
        }

        // Print a newline for each dimension (except the first one)
        if (current_dim != 0)
        {
            printf("\n");
        }
    }
}

void print_data(float *data, int shape[], int stride[], int ndim)
{
    int indices[ndim];
    print_data_ndim(data, shape, stride, indices, ndim, 0);
}