#include "ops.h"
#include "tensor.h"
#include "backops.h"
#include "helpers.h"

tensor_t *tensor_alloc(int size)
{
    tensor_t *tensor = (tensor_t *)malloc(sizeof(tensor_t));
    tensor->size = size;

    // shared pointers data & grad
    tensor->data = NULL;
    tensor->grad = NULL;

    // shape & stride
    tensor->shape = NULL;
    tensor->stride = NULL;

    // children
    tensor->child1 = NULL;
    tensor->child2 = NULL;

    // backward & forward
    tensor->forward = NULL;
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

    for (int i = ndim; i-- > 0; )
    {
        tensor->shape[i] = shape[i];
        tensor->stride[i] = (i == ndim - 1) ? 1 : tensor->stride[i + 1] * shape[i + 1];
    }

    return tensor;
}

tensor_t *tensor_init(int shape[], int ndim, bool requires_grad, float* (*op) (tensor_t*, tensor_t*))
{
    tensor_t *tensor = tensor_create(shape, ndim, requires_grad);
    tensor->forward = op;
    if (op == NULL) {
        tensor->data = smalloc(.size = tensor->size, .nmemb = sizeof(float), .kind = SHARED);
        set_data(tensor->data, 0., tensor->size);
    }
    // if (requires_grad) {
    //     tensor->grad = smalloc(.size = tensor->size, .nmemb = sizeof(float), .kind = SHARED);
    //     set_data(tensor->grad, 0., tensor->size);
    // }
    return tensor;
}

tensor_t *tensor(const float data[], int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_init(shape, ndim, requires_grad, NULL);
    memcpy(tensor->data, data, tensor->size * sizeof(float));
    return tensor;
}

tensor_t *tensor_rand(int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_init(shape, ndim, requires_grad, NULL);
    for (int i = 0; i < tensor->size; ++i)
    {
        tensor->data[i] = (float)rand() / (float)RAND_MAX;
    }
    return tensor;
}

tensor_t *tensor_zeros(int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_init(shape, ndim, requires_grad, NULL);
    set_data(tensor->data, 0., tensor->size);
    return tensor;
}

tensor_t *tensor_ones(int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_init(shape, ndim, requires_grad, NULL);
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
    ASSERT(size == tensor->size, "Size mismatch %d != %d", size, tensor->size);
    memcpy(tensor->data, data, size * sizeof(float));
}

void tensor_set_grad(tensor_t *tensor, float grad[], int size)
{
    ASSERT(size == tensor->size, "Size mismatch %d != %d", size, tensor->size);
    memcpy(tensor->grad, grad, size * sizeof(float));
}

bool tensor_same_shape(tensor_t *a, tensor_t *b, bool debug)
{
    bool same = is_same_shape(a->shape, b->shape, a->ndim, b->ndim);
    if (debug && !same)
    {
        print_metadata(a->shape, a->ndim);
        printf(" != ");
        print_metadata(b->shape, b->ndim);
        printf("\n");
    }
    return same;
}

bool tensor_equals(tensor_t *a, tensor_t *b, bool with_grad)
{
    if (!tensor_same_shape(a, b, false))
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
    // free data
    sfree(tensor->data);
    sfree(tensor->grad);
    // free metadata
    free(tensor->shape);
    free(tensor->stride);
    // free tensor
    free(tensor);
}

void tensor_print(tensor_t *tensor, flag_t flags)
{   
    printf("Tensor @ %p\n", (void*) tensor);
    printf("Shape:\t");
    print_metadata(tensor->shape, tensor->ndim);
    printf("\n");

    if (flags & PRINT_STRIDE)
    {
        printf("Stride:\t");
        print_metadata(tensor->stride, tensor->ndim);
        printf("\n");
    }

    if (flags & PRINT_CHILDREN) {
        printf("Children:\n");
        if (tensor->child1) {
            printf("\tChild 1: Tensor @ %p\n", (void*) tensor->child1);
        } else {
            printf("\tChild 1: NULL\n");
        }
        if (tensor->child2) {
            printf("\tChild 2: Tensor @ %p\n", (void*) tensor->child2);
        } else {
            printf("\tChild 2: NULL\n");
        }
    }

    if (flags & PRINT_DATA)
    {
        if (tensor->data)
        {
            printf("Data @ %p:\n", (void*) tensor->data);
            print_data(tensor->data, tensor->shape, tensor->stride, tensor->ndim);
        } else {
            printf("Data: NULL\n");
        }
    }

    if (flags & PRINT_GRAD) {
        if (tensor->grad)
        {
            printf("Grad @ %p:\n", (void*) tensor->grad);
            print_data(tensor->grad, tensor->shape, tensor->stride, tensor->ndim);
        } else {
            printf("Grad: NULL\n");
        }
    }
    printf("\n");
}

// BINARY OPS

tensor_t *tensor_add_tt(tensor_t *a, tensor_t *b)
{
    ASSERT(tensor_same_shape(a, b, true), "Add error :: Shape mismatch");
    tensor_t *out = tensor_init(a->shape, a->ndim, a->requires_grad || b->requires_grad, addt);
    out->child1 = a;
    out->child2 = b;
    if (out->requires_grad)
        out->backward = backward_add;

    return out;
}

tensor_t *tensor_add_tf(tensor_t *a, float b)
{
    tensor_t *out = tensor_init(a->shape, a->ndim, a->requires_grad, addt);
    out->child1 = a;
    out->child2 = tensor((float[]){b}, (int[]){1}, 1, false);
    if (out->requires_grad)
        out->backward = backward_add;

    return out;
}

tensor_t *tensor_add_ft(float a, tensor_t *b)
{
    return tensor_add_tf(b, a);
}

tensor_t *tensor_mul_tt(tensor_t *a, tensor_t *b)
{
    ASSERT(tensor_same_shape(a, b, true), "Mul error :: Shape mismatch");
    tensor_t *out = tensor_init(a->shape, a->ndim, a->requires_grad || b->requires_grad, mult);
    out->child1 = a;
    out->child2 = b;
    if (out->requires_grad)
        out->backward = backward_mul;

    return out;
}

tensor_t *tensor_mul_tf(tensor_t *a, float b)
{
    tensor_t *out = tensor_init(a->shape, a->ndim, a->requires_grad, mult);
    out->child1 = a;
    out->child2 = tensor((float[]){b}, (int[]){1}, 1, false);
    if (out->requires_grad)
        out->backward = backward_mul;

    return out;
}

tensor_t *tensor_mul_ft(float a, tensor_t *b)
{
    return tensor_mul_tf(b, a);
}

tensor_t *tensor_pow_tt(tensor_t *a, tensor_t *b)
{
    ASSERT(tensor_same_shape(a, b, true), "Pow error :: Shape mismatch");
    tensor_t *out = tensor_init(a->shape, a->ndim, a->requires_grad || b->requires_grad, powt);
    out->child1 = a;
    out->child2 = b;
    if (out->requires_grad)
        out->backward = backward_pow;

    return out;
}

tensor_t *tensor_pow_tf(tensor_t *a, float b)
{
    tensor_t *out = tensor_init(a->shape, a->ndim, a->requires_grad, powt);
    out->child1 = a;
    out->child2 = tensor((float[]){b}, (int[]){1}, 1, false);
    if (out->requires_grad)
        out->backward = backward_pow;

    return out;
}

tensor_t *tensor_pow_ft(float a, tensor_t *b)
{
    tensor_t *out = tensor_init(b->shape, b->ndim, b->requires_grad, powt);
    out->child1 = tensor((float[]){a}, (int[]){1}, 1, false);
    out->child2 = b;
    if (out->requires_grad)
        out->backward = backward_pow;

    return out;
}



tensor_t *tensor_reshape(tensor_t *tensor, int shape[], int ndim)
{
    int size = get_size(shape, ndim);
    ASSERT(size == tensor->size, "Size mismatch %d != %d", size, tensor->size);
    tensor_t *reshaped = tensor_create(shape, ndim, tensor->requires_grad);
    reshaped->data = sref(tensor->data);
    if (tensor->grad) reshaped->grad = sref(tensor->grad);
    for (int i = ndim; i-- > 0; )
    {
        reshaped->shape[i] = shape[i];
        reshaped->stride[i] = (i == ndim - 1) ? 1 : tensor->stride[i + 1] * shape[i + 1];
    };
    tensor->child1 = reshaped;
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

tensor_t *tensor_slice(tensor_t *tensor, slice_t range[])
{   
    // Compute new shape
    int shape[tensor->ndim];
    normalize_range(range, tensor->shape, tensor->ndim);
    for (int d = 0; d < tensor->ndim; d++)
    {
        shape[d] = abs(range[d].stop - range[d].start) / range[d].step + 
                  (abs(range[d].stop - range[d].start) % range[d].step != 0 ? 1 : 0);
    }

    // TODO: what about grad ?
    tensor_t *sliced = tensor_init(shape, tensor->ndim, tensor->requires_grad, NULL);

    // Fill sliced tensor with data
    int idx = 0;
    int src_idx[sliced->ndim];
    tensor_copy(sliced, tensor, &idx, src_idx, range, 0);

    return sliced;
}

tensor_t *tensor_cat(tensor_t *tensors[], int num_tensors, int axis)
{   
    int ndim = tensors[0]->ndim;
    ASSERT(axis >= 0 && axis < ndim, "Axis out of bounds: got %d", axis);

    // Compute new shape
    int shape[ndim];
    memcpy(shape, tensors[0]->shape, sizeof(shape));
    for (int i = 1; i < num_tensors; i++)
    {   
        ASSERT(tensors[i]->ndim == ndim, 
            "Number of dimensions must be the same for all tensors. Got %d and %d", ndim, tensors[i]->ndim);
        for (int d = 0; d < ndim; d++)
        {
            if (d != axis)
            {
                ASSERT(tensors[i]->shape[d] == shape[d], 
                    "Shape mismatch at axis %d: %d != %d", d, tensors[i]->shape[d], shape[d]);
            }
        }
        shape[axis] += tensors[i]->shape[axis];
    }

    // TODO: what about grad ?
    tensor_t *cated = tensor_init(shape, ndim, false, NULL);

    // Fill cated tensor with data
    int offset = 0;
    for (int i = 0; i < num_tensors; i++)
    {   
        int step = 0;
        int size = (axis == 0) ? tensors[i]->size : tensors[i]->stride[axis-1];
        int n = tensors[i]->size / size;
        for (int j = 0; j < n; j++) {
            memcpy(cated->data + offset + step, tensors[i]->data + j * size, size * sizeof(float));
            step += cated->stride[axis - 1];
        }
        offset += size;
    }
    return cated;
}

void tensor_copy(tensor_t *dst, tensor_t *src, int *dst_idx, int *src_idx, slice_t *range, int dim)
{
    if (dim == src->ndim)
    {
        dst->data[(*dst_idx)++] = src->data[get_index(src_idx, src->stride, src->ndim)];
    }
    else
    {
        for (int i = range[dim].start; i < range[dim].stop; i += range[dim].step)
        {
            src_idx[dim] = i;
            tensor_copy(dst, src, dst_idx, src_idx, range, dim + 1);
        }
    }
}

void tensor_forward(tensor_t *tensor)
{   
    if (tensor->forward && tensor->data == NULL)
    {   
        tensor_forward(tensor->child1);
        tensor_forward(tensor->child2);
        tensor->data = tensor->forward(tensor->child1, tensor->child2);
    }
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