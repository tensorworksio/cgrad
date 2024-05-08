#include "ops.h"
#include "tensor.h"
#include "helpers.h"

// ALLOC OPS
tensor_t *tensor_alloc(int size)
{
    tensor_t *tensor = (tensor_t *)malloc(sizeof(tensor_t));
    tensor->size = size;

    // shared pointers data & grad
    tensor->data = NULL;
    tensor->grad = NULL;

    // shape, stride & range
    tensor->shape = NULL;
    tensor->range = NULL;
    tensor->stride = NULL;

    // parents
    tensor->n_parents = 0;
    tensor->parents = NULL;

    // children
    tensor->n_children = 0;
    tensor->children = NULL;

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
    tensor->range = (slice_t *)malloc(sizeof(slice_t) * ndim);

    for (int i = ndim; i-- > 0;)
    {
        tensor->shape[i] = shape[i];
        tensor->range[i] = (slice_t){0, shape[i], 1};
        tensor->stride[i] = (i == ndim - 1) ? 1 : tensor->stride[i + 1] * shape[i + 1];
    }

    return tensor;
}

tensor_t *tensor_init(int shape[], int ndim, bool requires_grad, void (*op)(tensor_t *))
{
    tensor_t *tensor = tensor_create(shape, ndim, requires_grad);
    tensor->forward = op;
    if (op == NULL)
    {
        tensor->data = smalloc(.size = tensor->size, .nmemb = sizeof(float), .kind = SHARED);
        set_data(tensor->data, 0., tensor->size);
    }
    return tensor;
}

void tensor_link(tensor_t *child, tensor_t *parent)
{
    for (int i = 0; i < child->n_parents; i++)
    {
        ASSERT(child->parents[i] != parent, "Parent already linked to child.");
    }

    for (int i = 0; i < parent->n_children; i++)
    {
        ASSERT(parent->children[i] != child, "Child already linked to parent.");
    }

    child->n_parents++;
    child->parents = realloc(child->parents, child->n_parents * sizeof(tensor_t *));
    child->parents[child->n_parents - 1] = parent;

    parent->n_children++;
    parent->children = realloc(parent->children, parent->n_children * sizeof(tensor_t *));
    parent->children[parent->n_children - 1] = child;
}

// DESTRUCT OPS
void tensor_unlink(tensor_t *child, tensor_t *parent)
{
    bool linked = false;
    for (int i = 0; i < child->n_parents; i++)
    {
        if (child->parents[i] != parent)
            continue;

        linked = true;
        for (int j = i; j < child->n_parents - 1; j++)
        {
            child->parents[j] = child->parents[j + 1];
        }
        child->n_parents--;
        child->parents = realloc(child->parents, child->n_parents * sizeof(tensor_t *));
        break;
    }

    ASSERT(linked, "Parent not linked to child.");

    linked = false;
    for (int i = 0; i < parent->n_children; i++)
    {
        if (parent->children[i] != child)
            continue;

        linked = true;
        for (int j = i; j < parent->n_children - 1; j++)
        {
            parent->children[j] = parent->children[j + 1];
        }
        parent->n_children--;
        parent->children = realloc(parent->children, parent->n_children * sizeof(tensor_t *));
        break;
    }

    ASSERT(linked, "Child not linked to parent.");
}

void tensor_free(tensor_t *tensor, bool recursive)
{
    if (recursive)
    {
        int n_parents = tensor->n_parents;
        // unlink & free parents
        for (int i = 0; i < n_parents; ++i)
        {
            // always index 0 because of realloc in tensor_unlink
            tensor_free(tensor->parents[0], recursive);
        }
    }
    // unlink children
    int n_children = tensor->n_children;
    for (int i = 0; i < n_children; ++i)
    {
        // always index 0 because of realloc in tensor_unlink
        tensor_unlink(tensor->children[0], tensor);
    }
    // free data
    sfree(tensor->data);
    sfree(tensor->grad);
    // free metadata
    free(tensor->shape);
    free(tensor->range);
    free(tensor->stride);
    // free tensors
    free(tensor->parents);
    free(tensor->children);
    free(tensor);
}

// INIT OPS
tensor_t *tensor(const float data[], int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_init(shape, ndim, requires_grad, NULL);
    memcpy(tensor->data, data, tensor->size * sizeof(float));
    return tensor;
}

tensor_t *tensor_copy(tensor_t *tensor, bool with_grad)
{
    tensor_t *out = tensor_init(tensor->shape, tensor->ndim, tensor->requires_grad, NULL);
    memcpy(out->data, tensor->data, tensor->size * sizeof(float));
    memcpy(out->stride, tensor->stride, tensor->ndim * sizeof(int));
    memcpy(out->range, tensor->range, tensor->ndim * sizeof(slice_t));
    if (with_grad && tensor->requires_grad)
    {
        memcpy(out->grad, tensor->grad, tensor->size * sizeof(float));
    }
    return out;
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
    if (tensor->data == NULL)
    {
        tensor->data = smalloc(.size = size, .nmemb = sizeof(float), .kind = SHARED);
    }
    memcpy(tensor->data, data, size * sizeof(float));
}

void tensor_set_grad(tensor_t *tensor, float grad[], int size)
{
    ASSERT(size == tensor->size, "Size mismatch %d != %d", size, tensor->size);
    if (tensor->grad == NULL)
    {
        tensor->grad = smalloc(.size = size, .nmemb = sizeof(float), .kind = SHARED);
    }
    memcpy(tensor->grad, grad, size * sizeof(float));
}

// COMPARISON OPS
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
    bool result = true;
    iterator_t it_a = tensor_iterator(a);
    iterator_t it_b = tensor_iterator(b);

    do
    {
        if (!tensor_same_shape(a, b, false))
        {
            result = false;
            break;
        }
        if (!is_equal_data(a->data, b->data, &it_a, &it_b))
        {
            result = false;
            break;
        }
        if (with_grad && a->requires_grad != b->requires_grad)
        {
            result = false;
            break;
        }
        if (a->requires_grad && b->requires_grad && !is_equal_data(a->grad, b->grad, &it_a, &it_b))
        {
            result = false;
            break;
        }
    } while (0);

    iterator_free(&it_a);
    iterator_free(&it_b);

    return result;
}

// UNARY OPS
tensor_t *tensor_neg(tensor_t *a)
{
    return tensor_mul_tf(a, -1.0);
}

tensor_t *tensor_exp(tensor_t *a)
{
    return tensor_pow_ft(expf(1.0), a);
}

tensor_t *tensor_relu(tensor_t *a)
{
    tensor_t *out = tensor_init(a->shape, a->ndim, a->requires_grad, forward_relu);
    tensor_link(out, a);
    if (out->requires_grad)
        out->backward = backward_relu;

    return out;
}

// BINARY OPS
tensor_t *tensor_add(tensor_t *a, tensor_t *b)
{
    return tensor_add_tt(a, b);
}

tensor_t *tensor_sub(tensor_t *a, tensor_t *b)
{
    return tensor_sub_tt(a, b);
}

tensor_t *tensor_mul(tensor_t *a, tensor_t *b)
{
    return tensor_mul_tt(a, b);
}

tensor_t *tensor_div(tensor_t *a, tensor_t *b)
{
    return tensor_div_tt(a, b);
}

tensor_t *tensor_pow(tensor_t *a, tensor_t *b)
{
    return tensor_pow_tt(a, b);
}

tensor_t *tensor_add_tt(tensor_t *a, tensor_t *b)
{
    ASSERT(tensor_same_shape(a, b, true), "Add error :: Shape mismatch");
    tensor_t *out = tensor_init(a->shape, a->ndim, a->requires_grad || b->requires_grad, forward_add);
    tensor_link(out, a);
    tensor_link(out, b);
    if (out->requires_grad)
        out->backward = backward_add;

    return out;
}

tensor_t *tensor_add_tf(tensor_t *a, float b)
{
    tensor_t *out = tensor_init(a->shape, a->ndim, a->requires_grad, forward_add);
    tensor_link(out, a);
    tensor_link(out, tensor((float[]){b}, (int[]){1}, 1, false));
    if (out->requires_grad)
        out->backward = backward_add;

    return out;
}

tensor_t *tensor_add_ft(float a, tensor_t *b)
{
    return tensor_add_tf(b, a);
}

tensor_t *tensor_sub_tt(tensor_t *a, tensor_t *b)
{
    return tensor_add(a, tensor_neg(b));
}

tensor_t *tensor_sub_tf(tensor_t *a, float b)
{
    return tensor_add_tf(a, -b);
}

tensor_t *tensor_sub_ft(float a, tensor_t *b)
{
    return tensor_add_ft(a, tensor_neg(b));
}

tensor_t *tensor_mul_tt(tensor_t *a, tensor_t *b)
{
    ASSERT(tensor_same_shape(a, b, true), "Mul error :: Shape mismatch");
    tensor_t *out = tensor_init(a->shape, a->ndim, a->requires_grad || b->requires_grad, forward_mul);
    tensor_link(out, a);
    tensor_link(out, b);
    if (out->requires_grad)
        out->backward = backward_mul;

    return out;
}

tensor_t *tensor_mul_tf(tensor_t *a, float b)
{
    tensor_t *out = tensor_init(a->shape, a->ndim, a->requires_grad, forward_mul);
    tensor_link(out, a);
    tensor_link(out, tensor((float[]){b}, (int[]){1}, 1, false));
    if (out->requires_grad)
        out->backward = backward_mul;

    return out;
}

tensor_t *tensor_mul_ft(float a, tensor_t *b)
{
    return tensor_mul_tf(b, a);
}

tensor_t *tensor_div_tt(tensor_t *a, tensor_t *b)
{
    return tensor_mul(a, tensor_pow_tf(b, -1.0));
}

tensor_t *tensor_div_tf(tensor_t *a, float b)
{
    return tensor_mul_tf(a, 1.0 / b);
}

tensor_t *tensor_div_ft(float a, tensor_t *b)
{
    return tensor_mul_ft(a, tensor_pow_tf(b, -1.0));
}

tensor_t *tensor_pow_tt(tensor_t *a, tensor_t *b)
{
    ASSERT(tensor_same_shape(a, b, true), "Pow error :: Shape mismatch");
    tensor_t *out = tensor_init(a->shape, a->ndim, a->requires_grad || b->requires_grad, forward_pow);
    tensor_link(out, a);
    tensor_link(out, b);
    if (out->requires_grad)
        out->backward = backward_pow;

    return out;
}

tensor_t *tensor_pow_tf(tensor_t *a, float b)
{
    tensor_t *out = tensor_init(a->shape, a->ndim, a->requires_grad, forward_pow);
    tensor_link(out, a);
    tensor_link(out, tensor((float[]){b}, (int[]){1}, 1, false));
    if (out->requires_grad)
        out->backward = backward_pow;

    return out;
}

tensor_t *tensor_pow_ft(float a, tensor_t *b)
{
    tensor_t *out = tensor_init(b->shape, b->ndim, b->requires_grad, forward_pow);
    tensor_link(out, tensor((float[]){a}, (int[]){1}, 1, false));
    tensor_link(out, b);
    if (out->requires_grad)
        out->backward = backward_pow;

    return out;
}

// REDUCE OPS
tensor_t *tensor_reduce(tensor_t *tensor, int axes[], int n_axes, tensor_t *(*op)(tensor_t *, tensor_t *))
{
    ASSERT(n_axes > 0, "Number of axes must be greater than 0.");
    tensor_t *out = tensor_reduce_axis(tensor, axes[0], op);
    for (int i = 1; i < n_axes; i++)
    {
        out = tensor_reduce_axis(out, axes[i], op);
    }
    return out;
}

tensor_t *tensor_reduce_axis(tensor_t *tensor, int axis, tensor_t *(*op)(tensor_t *, tensor_t *))
{
    slice_t *range = malloc(tensor->ndim * sizeof(slice_t));
    memcpy(range, tensor->range, tensor->ndim * sizeof(slice_t));

    int n = tensor->shape[axis];
    range[axis] = SLICE_ONE(0);
    tensor_t *out = tensor_slice(tensor, range);

    for (int i = 1; i < n; i++)
    {
        range[axis] = SLICE_ONE(i);
        out = op(out, tensor_slice(tensor, range));
    }
    free(range);
    return out;
}

tensor_t *tensor_sum_axes(tensor_t *tensor, int axes[], int n_axes)
{
    return tensor_reduce(tensor, axes, n_axes, tensor_add);
}

tensor_t *tensor_sum(tensor_t *tensor)
{
    int axes[tensor->ndim];
    for (int i = 0; i < tensor->ndim; i++)
    {
        axes[i] = i;
    }
    return tensor_sum_axes(tensor, axes, tensor->ndim);
}

// MOVEMENT OPS
tensor_t *tensor_reshape(tensor_t *tensor, int shape[], int ndim)
{
    int size = get_size(shape, ndim);
    ASSERT(size == tensor->size, "Size mismatch %d != %d", size, tensor->size);
    tensor_t *out = tensor_init(shape, ndim, tensor->requires_grad, forward_ref);
    tensor_link(out, tensor);
    if (out->requires_grad)
        out->backward = backward_ref;

    return out;
}

tensor_t *tensor_transpose(tensor_t *tensor, int axis1, int axis2)
{
    ASSERT(tensor->ndim + axis1 >= 0 && axis1 < tensor->ndim, "Axis1 out of bounds: got %d", axis1);
    ASSERT(tensor->ndim + axis2 >= 0 && axis2 < tensor->ndim, "Axis2 out of bounds: got %d", axis2);

    axis1 = (axis1 < 0) ? tensor->ndim + axis1 : axis1;
    axis2 = (axis2 < 0) ? tensor->ndim + axis2 : axis2;

    tensor_t *out = tensor_init(tensor->shape, tensor->ndim, tensor->requires_grad, forward_ref);
    swap_int(out->shape, axis1, axis2);
    swap_int(out->stride, axis1, axis2);
    swap_slice(out->range, axis1, axis2);

    tensor_link(out, tensor);
    if (out->requires_grad)
        out->backward = backward_ref;

    return out;
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

    tensor_t *out = tensor_init(shape, tensor->ndim, tensor->requires_grad, forward_slice);
    tensor_link(out, tensor);
    memcpy(out->range, range, sizeof(slice_t) * out->ndim);
    // memcpy(out->stride, tensor->stride, sizeof(int) * out->ndim);

    if (out->requires_grad)
        out->backward = backward_copy;

    return out;
}

tensor_t *tensor_cat(tensor_t *tensors[], int num_tensors, int axis)
{
    int ndim = tensors[0]->ndim;
    ASSERT(ndim + axis >= 0 && axis < ndim, "Axis out of bounds: got %d", axis);

    // Compute new shape
    int shape[ndim];
    axis = (axis < 0) ? ndim + axis : axis;
    bool requires_grad = tensors[0]->requires_grad;
    memcpy(shape, tensors[0]->shape, ndim * sizeof(int));
    for (int i = 1; i < num_tensors; i++)
    {
        ASSERT(tensors[i]->ndim == ndim,
               "Number of dimensions must be the same for all tensors. Got %d and %d", ndim, tensors[i]->ndim);
        requires_grad = requires_grad || tensors[i]->requires_grad;
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

    // Compute new range
    int end;
    int start = tensors[0]->shape[axis];
    for (int i = 1; i < num_tensors; i++)
    {
        end = start + tensors[i]->shape[axis];
        tensors[i]->range[axis] = (slice_t){start, end, 1};
        start = end;
    }

    tensor_t *out = tensor_init(shape, ndim, requires_grad, forward_cat);
    for (int i = 0; i < num_tensors; i++)
    {
        tensor_link(out, tensors[i]);
        memcpy(tensors[i]->stride, out->stride, sizeof(int) * ndim);
    }
    if (out->requires_grad)
        out->backward = backward_ref;

    return out;
}

// FORCING OPS
void tensor_forward(tensor_t *tensor)
{
    if (tensor->forward == NULL)
    {
        log_debug("Node %p has no forward function.\n", (void *)tensor);
        return;
    }
    // TODO: check on data = NULL should be removed and replaced with a flag
    // because we could modify data even if it's already initialized
    if (tensor->data == NULL)
    {
        for (int i = 0; i < tensor->n_parents; ++i)
        {
            tensor_forward(tensor->parents[i]);
        }
        tensor->forward(tensor);
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
    if (tensor->backward == NULL)
    {
        log_warn("Node %p has no backward function.\n", (void *)tensor);
        return;
    }
    if (tensor->parents == NULL)
    {
        log_warn("Node %p has no parents.", (void *)tensor);
        return;
    }
    tensor_forward(tensor);
    // TODO: check on grad = NULL should be removed and replaced with a flag
    // because we could modify grad even if it's already initialized
    if (tensor->grad == NULL)
    {
        tensor->grad = smalloc(.size = tensor->size, .nmemb = sizeof(float), .kind = SHARED);
        tensor_init_grad(tensor);
        tensor->backward(tensor);
    }
}

void tensor_print(tensor_t *tensor, print_flag_t flags)
{
    printf("Tensor @ %p\n", (void *)tensor);
    printf("Shape:\t");
    print_metadata(tensor->shape, tensor->ndim);
    printf("\n");

    if (flags & PRINT_STRIDE)
    {
        printf("Stride:\t");
        print_metadata(tensor->stride, tensor->ndim);
        printf("\n");
    }

    if (flags & PRINT_CHILDREN)
    {
        if (!tensor->parents)
        {
            printf("parents: NULL\n");
        }
        else
        {
            printf("parents:\n");
            for (int i = 0; i < tensor->n_parents; ++i)
            {
                printf("\tChild %d: Tensor @ %p\n", i + 1, (void *)tensor->parents[i]);
            }
        }
    }

    if (flags & PRINT_DATA)
    {
        if (tensor->data)
        {
            printf("Data @ %p:\n", (void *)tensor->data);
            iterator_t it = tensor_iterator(tensor);
            print_data(tensor->data, &it);
            iterator_free(&it);
        }
        else
        {
            printf("Data: NULL\n");
        }
    }

    if (flags & PRINT_GRAD)
    {
        if (tensor->grad)
        {
            printf("Grad @ %p:\n", (void *)tensor->grad);
            iterator_t it = tensor_iterator(tensor);
            print_data(tensor->grad, &it);
            iterator_free(&it);
        }
        else
        {
            printf("Grad: NULL\n");
        }
    }
    printf("\n");
}

iterator_t tensor_iterator(tensor_t *tensor)
{
    return iterator(tensor->range, tensor->stride, tensor->ndim);
}