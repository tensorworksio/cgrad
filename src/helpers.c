#include "helpers.h"

size_t
get_size (int shape[], size_t ndim)
{
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++)
    {
        size *= shape[i];
    }
    return size;
}

int
get_index (int coords[], int stride[], size_t ndim)
{
    int index = 0;
    for (size_t i = 0; i < ndim; i++)
    {
        index += coords[i] * stride[i];
    }
    return index;
}

void
set_data (float *data, float value, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        data[i] = value;
    }
}

bool
is_same_shape (int shape_a[], int shape_b[], size_t ndim_a, size_t ndim_b)
{
    if (ndim_a != ndim_b)
        return false;

    for (size_t i = 0; i < ndim_a; i++)
    {
        if (shape_a[i] != shape_b[i])
            return false;
    }
    return true;
}

bool
is_equal_data (float *data_a, float *data_b, size_t size)
{
    if (data_a == NULL || data_b == NULL)
        return false;
    for (size_t i = 0; i < size; i++)
    {
        if (fabs (data_a[i] - data_b[i]) > EPSILON)
            return false;
    }
    return true;
}

void
normalize_range (slice_t range[], int shape[], size_t ndim)
{
    for (size_t d = 0; d < ndim; d++)
    {
        ASSERT (range[d].step > 0, "Slice step must be positive: got %d", range[d].step);
        ASSERT (abs (range[d].start) <= shape[d],
                "Index %d is out of bounds for dimension %d with size %d", range[d].start, d,
                shape[d]);
        ASSERT (abs (range[d].stop) <= shape[d],
                "Index %d is out of bounds for dimension %d with size %d", range[d].stop, d,
                shape[d]);

        // if start/stop is negative, add shape to it
        range[d].start = (range[d].start < 0) ? range[d].start + shape[d] : range[d].start;
        range[d].stop  = (range[d].stop < 0) ? range[d].stop + shape[d] + 1 : range[d].stop;

        ASSERT (range[d].start <= range[d].stop,
                "Slice start must be less than or equal to slice stop");
    }
}
void
print_metadata (int data[], size_t ndim)
{
    printf ("[");
    for (size_t i = 0; i < ndim - 1; i++)
    {
        printf ("%d, ", data[i]);
    }
    printf ("%d]", data[ndim - 1]);
}

void
print_data_ndim (float *data, int shape[], int stride[], int indices[], size_t ndim, size_t dim)
{
    if (dim == ndim)
    {
        int index = get_index (indices, stride, ndim);
        printf ("%f ", data[index]);
        return;
    }

    for (indices[dim] = 0; indices[dim] < shape[dim]; indices[dim]++)
    {
        print_data_ndim (data, shape, stride, indices, ndim, dim + 1);
    }
    printf ("\n");
}

void
print_data (float *data, int shape[], int stride[], size_t ndim)
{
    int indices[(size_t) ndim];
    for (int i = 0; i < (int) ndim; i++)
        indices[i] = 0;
    print_data_ndim (data, shape, stride, indices, ndim, 0);
}

void
build_topo (tensor_t *root, tensor_t ***list, int *count, int *capacity)
{
    if (root->op == NULL || root->op->visited)
    {
        return;
    }

    root->op->visited = true;

    for (size_t i = 0; i < root->n_children; ++i)
    {
        build_topo (root->children[i], list, count, capacity);
    }

    if (*count == *capacity)
    {
        *capacity *= 2;
        *list = realloc (*list, *capacity * sizeof (tensor_t *));
    }

    (*list)[(*count)++] = root;
}