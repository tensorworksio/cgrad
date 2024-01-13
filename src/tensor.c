#include "tensor.h"
#include "backops.h"
#include "helpers.h"

tensor_t *tensor_alloc(int size)
{
    tensor_t *tensor = (tensor_t *)malloc(sizeof(tensor_t));
    tensor->data = (float *)calloc(size, sizeof(float));
    tensor->grad = NULL; // grad is only allocated if needed
    tensor->size = size;
    tensor->child1 = NULL;
    tensor->child2 = NULL;
    tensor->backward = NULL;
    return tensor;
}

tensor_t *tensor_create(int shape[], int ndim, bool requires_grad)
{
    int size = get_size(shape, ndim);
    tensor_t *tensor = tensor_alloc(size);
    tensor->ndim = ndim;
    tensor->requires_grad = requires_grad;
    tensor->shape = (int *)malloc(sizeof(int) * ndim);
    memcpy(tensor->shape, shape, sizeof(int) * ndim);
    if (requires_grad)
        tensor->grad = (float *)calloc(size, sizeof(float));
    return tensor;
}

tensor_t *tensor(const float data[], int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_create(shape, ndim, requires_grad);
    memcpy(tensor->data, data, sizeof(float) * tensor->size);
    return tensor;
}

tensor_t *tensor_rand(int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_create(shape, ndim, requires_grad);
    for (int i = 0; i < tensor->size; ++i)
    {
        tensor->data[i] = (float)rand() / (float)RAND_MAX;
    }
    return tensor;
}

tensor_t *tensor_zeros(int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_create(shape, ndim, requires_grad);
    set_data(tensor->data, 0., tensor->size);
    return tensor;
}

void tensor_zero_grad(tensor_t *tensor)
{
    set_data(tensor->grad, 0., tensor->size);
}

tensor_t *tensor_ones(int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_create(shape, ndim, requires_grad);
    set_data(tensor->data, 1., tensor->size);
    return tensor;
}

void tensor_init_grad(tensor_t *tensor)
{
    set_data(tensor->grad, 1., tensor->size);
}

void tensor_set_data(tensor_t *tensor, float data[], int size)
{
    assert(size == tensor->size && "Size mismatch");
    memcpy(tensor->data, data, size * sizeof(float));
}

void tensor_set_grad(tensor_t *tensor, float grad[], int size)
{
    assert(size == tensor->size && "Size mismatch");
    memcpy(tensor->grad, grad, size * sizeof(float));
}

bool tensor_same_shape(tensor_t *a, tensor_t *b)
{
    return is_same_shape(a->shape, b->shape, a->ndim, b->ndim);
}

bool tensor_equals(tensor_t *a, tensor_t *b, bool with_grad)
{
    if (!tensor_same_shape(a, b))
        return false;
    if (!is_equal_data(a->data, b->data, a->size))
        return false;
    if (with_grad && a->requires_grad != b->requires_grad)
        return false;
    if (a->requires_grad && b->requires_grad && !is_equal_data(a->grad, b->grad, a->size))
        return false;
    return true;
}

void tensor_free(tensor_t *tensor, bool recursive)
{
    if (recursive)
    {
        if (tensor->child1)
            tensor_free(tensor->child1, recursive);
        if (tensor->child2)
            tensor_free(tensor->child2, recursive);
    }

    free(tensor->data);
    free(tensor->grad);
    free(tensor->shape);
    free(tensor);
}

void tensor_print(tensor_t *tensor)
{
    printf("DATA\n");
    print_data(tensor->data, tensor->shape, tensor->ndim);
    printf("\n");
    if (tensor->requires_grad)
    {
        printf("GRAD\n");
        print_data(tensor->grad, tensor->shape, tensor->ndim);
        printf("\n");
    }
}

void tensor_fill(tensor_t *dst, tensor_t *src, int *dst_idx, int *src_idx, slice_t *ranges, int dim)
{
    if (dim == src->ndim)
    {
        dst->data[(*dst_idx)++] = src->data[get_index(src->shape, src_idx, src->ndim)];
    }
    else
    {
        for (int i = ranges[dim].start; i < ranges[dim].stop; i += ranges[dim].step)
        {
            src_idx[dim] = i;
            tensor_fill(dst, src, dst_idx, src_idx, ranges, dim + 1);
        }
    }
}

tensor_t *tensor_slice(tensor_t *tensor, slice_t ranges[], int ndim)
{
    assert(ndim == tensor->ndim && "Number of ranges must be equal to the number of dimensions");
    int shape[ndim];
    // set the range for each dimension
    set_ranges(ranges, tensor->shape, ndim);
    // set the shape of the new tensor
    set_shape(shape, ranges, ndim);
    // Create the new tensor
    tensor_t *out = tensor_create(shape, ndim, tensor->requires_grad);
    // Fill the new tensor with the sliced data
    int idx[ndim];
    int out_idx = 0;
    tensor_fill(out, tensor, &out_idx, idx, ranges, 0);
    return out;
}

void tensor_backward(tensor_t *tensor)
{
    if (!tensor->requires_grad)
    {
        log_error("Cannot perform backward on tensor that has no grad.\n");
        tensor_free(tensor, true);
        exit(EXIT_FAILURE);
    }
    if (tensor->size != 1)
    {
        log_error("Backward operation only supported for scalar tensors.\n");
        tensor_free(tensor, true);
        exit(EXIT_FAILURE);
    }
    tensor_init_grad(tensor);
    backward(tensor);
}