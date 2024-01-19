#include "tensor.h"
#include "backops.h"
#include "helpers.h"

tensor_t *tensor_alloc(int size)
{
    tensor_t *tensor = (tensor_t *)malloc(sizeof(tensor_t));
    tensor->size = size;

    // data & grad are only allocated if needed
    tensor->data = NULL;
    tensor->grad = NULL;

    // no children by default
    tensor->child1 = NULL;
    tensor->child2 = NULL;
    tensor->backward = NULL;
    return tensor;
}

tensor_t *tensor_create(int shape[], int ndim, bool requires_grad)
{
    int size = get_size(shape, ndim);
    tensor_t *tensor = tensor_alloc(size);

    // tensor skeleton
    tensor->ndim = ndim;
    tensor->requires_grad = requires_grad;
    tensor->shape = (int *)malloc(sizeof(int) * ndim);
    tensor->stride = (int *)malloc(sizeof(int) * ndim);

    for (int i = ndim - 1; i >= 0; i--)
    {
        tensor->shape[i] = shape[i];
        tensor->stride[i] = (i == ndim - 1) ? 1 : tensor->stride[i + 1] * shape[i + 1];
    }

    return tensor;
}

tensor_t *tensor_init(int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_create(shape, ndim, requires_grad);
    tensor->data = (float *)calloc(tensor->size, sizeof(float));
    if (requires_grad)
        tensor->grad = (float *)calloc(tensor->size, sizeof(float));
    return tensor;
}

tensor_t *tensor(const float data[], int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_init(shape, ndim, requires_grad);
    memcpy(tensor->data, data, tensor->size * sizeof(float));
    return tensor;
}

tensor_t *tensor_rand(int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_init(shape, ndim, requires_grad);
    for (int i = 0; i < tensor->size; ++i)
    {
        tensor->data[i] = (float)rand() / (float)RAND_MAX;
    }
    return tensor;
}

tensor_t *tensor_zeros(int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_init(shape, ndim, requires_grad);
    set_data(tensor->data, 0., tensor->size);
    return tensor;
}

tensor_t *tensor_ones(int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_init(shape, ndim, requires_grad);
    set_data(tensor->data, 1., tensor->size);
    return tensor;
}

void tensor_zero_grad(tensor_t *tensor)
{
    set_data(tensor->grad, 0., tensor->size);
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
    free(tensor->stride);
    free(tensor);
}

void tensor_print(tensor_t *tensor)
{
    printf("DATA\n");
    print_data(tensor->data, tensor->shape, tensor->stride, tensor->ndim);
    printf("\n");
    if (tensor->requires_grad)
    {
        printf("GRAD\n");
        print_data(tensor->grad, tensor->shape, tensor->stride, tensor->ndim);
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

tensor_t *tensor_reshape(tensor_t *tensor, int shape[], int ndim)
{
    assert(tensor->size == get_size(shape, ndim) && "Size mismatch");
    tensor_t *reshaped = tensor_alloc(tensor->size);
    reshaped->data = tensor->data;
    reshaped->grad = tensor->grad;
    for (int i = ndim - 1; i >= 0; i--)
    {
        reshaped->shape[i] = shape[i];
        reshaped->stride[i] = (i == ndim - 1) ? 1 : tensor->stride[i + 1] * shape[i + 1];
    };
    return reshaped;
}

tensor_t *tensor_transpose(tensor_t *tensor, int axis1, int axis2)
{
    int shape[tensor->ndim];
    // set shape
    memcpy(shape, tensor->shape, sizeof(shape));
    shape[axis1] = tensor->shape[axis2];
    shape[axis2] = tensor->shape[axis1];

    tensor_t *transposed = tensor_reshape(tensor, shape, tensor->ndim);
    // set stride
    memcpy(transposed->stride, tensor->stride, tensor->ndim * sizeof(int));
    transposed->stride[axis1] = tensor->stride[axis2];
    transposed->stride[axis2] = tensor->stride[axis1];

    return transposed;
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
    tensor_t *out = tensor_init(shape, ndim, tensor->requires_grad);
    // Fill the new tensor with the sliced data
    int idx[ndim];
    int out_idx = 0;
    tensor_fill(out, tensor, &out_idx, idx, ranges, 0);
    return out;
}

tensor_t *tensor_cat(tensor_t *tensors[], int num_tensors, int axis)
{
    // only works when we concateneta along axis 0
    // we should simply transpose the tensors and concatenate along axis 0 then transpose back
    int ndim = tensors[0]->ndim;
    int shape[ndim];
    memcpy(shape, tensors[0]->shape, sizeof(shape));
    for (int i = 1; i < num_tensors; i++)
    {
        shape[axis] += tensors[i]->shape[axis];
    }

    tensor_t *c = tensor_init(shape, ndim, false);
    int dst_idx = 0;
    int src_idx[ndim];

    for (int i = 0; i < num_tensors; i++)
    {
        slice_t ranges[ndim];
        for (int j = 0; j < ndim; j++)
        {
            ranges[j] = (slice_t){0, tensors[i]->shape[j], 1};
        }
        tensor_fill(c, tensors[i], &dst_idx, src_idx, ranges, axis);
    }

    return c;
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