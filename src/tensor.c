#include "tensor.h"
#include "forward.h"
#include "helpers.h"

// ALLOC OPS
void
tensor_destructor (void *ptr, void *meta)
{
    tensor_t *tensor = (tensor_t *) ptr;

    for (size_t i = 0; i < tensor->n_children; ++i)
    {
        sfree (tensor->children[i]);
    }

    // Free metadata only at actual deallocation time
    free (tensor->shape);
    free (tensor->stride);
    free (tensor->children);

    // Free operator
    sfree (tensor->op);

    // Free data & grad (shared)
    sfree (tensor->data);
    sfree (tensor->grad);
}

tensor_t *
tensor_create (int shape[], size_t ndim, bool requires_grad)
{
    size_t    size   = get_size (shape, ndim);
    tensor_t *tensor = shared_ptr (tensor_t,
                                   { .size          = size,
                                     .ndim          = ndim,
                                     .requires_grad = requires_grad,
                                     .data          = NULL,
                                     .grad          = NULL,
                                     .shape         = NULL,
                                     .stride        = NULL,
                                     .n_children    = 0,
                                     .children      = NULL,
                                     .op            = NULL },
                                   tensor_destructor);

    tensor->shape  = (int *) malloc (sizeof (int) * ndim);
    tensor->stride = (int *) malloc (sizeof (int) * ndim);

    for (size_t i = ndim; i-- > 0;)
    {
        tensor->shape[i]  = shape[i];
        tensor->stride[i] = (i == ndim - 1) ? 1 : tensor->stride[i + 1] * shape[i + 1];
    }

    return tensor;
}

tensor_t *
tensor_init (int shape[], size_t ndim, bool requires_grad, op_t *op)
{
    tensor_t *tensor = tensor_create (shape, ndim, requires_grad);
    tensor->op       = op;
    if (op == NULL)
    {
        tensor->data = smalloc (.nmemb = tensor->size, .size = sizeof (float), .kind = SHARED);
        set_data (tensor->data, 0., tensor->size);
    }
    return tensor;
}

void
tensor_add_child (tensor_t *tensor, tensor_t *child)
{
    tensor->n_children++;
    tensor->children = realloc (tensor->children, tensor->n_children * sizeof (tensor_t *));
    tensor->children[tensor->n_children - 1] = sref (child);
}

// INIT OPS
tensor_t *
tensor (const float data[], int shape[], size_t ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_init (shape, ndim, requires_grad, NULL);
    memcpy (tensor->data, data, tensor->size * sizeof (float));
    return tensor;
}

tensor_t *
tensor_rand (int shape[], size_t ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_init (shape, ndim, requires_grad, NULL);
    for (int i = 0; i < tensor->size; ++i)
    {
        tensor->data[i] = (float) rand () / (float) RAND_MAX;
    }
    return tensor;
}

tensor_t *
tensor_zeros (int shape[], size_t ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_init (shape, ndim, requires_grad, NULL);
    set_data (tensor->data, 0., tensor->size);
    return tensor;
}

tensor_t *
tensor_ones (int shape[], size_t ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_init (shape, ndim, requires_grad, NULL);
    set_data (tensor->data, 1., tensor->size);
    return tensor;
}

void
tensor_zero_grad (tensor_t *tensor)
{
    set_data (tensor->grad, 0., tensor->size);
}

void
tensor_init_grad (tensor_t *tensor)
{
    set_data (tensor->grad, 1., tensor->size);
}

void
tensor_set_data (tensor_t *tensor, float data[], size_t size)
{
    ASSERT (size == tensor->size, "Size mismatch %d != %d", size, tensor->size);
    if (tensor->data == NULL)
    {
        tensor->data = smalloc (.nmemb = size, .size = sizeof (float), .kind = SHARED);
    }
    memcpy (tensor->data, data, size * sizeof (float));
}

void
tensor_set_grad (tensor_t *tensor, float grad[], size_t size)
{
    ASSERT (size == tensor->size, "Size mismatch %d != %d", size, tensor->size);
    if (tensor->grad == NULL)
    {
        tensor->grad = smalloc (.nmemb = size, .size = sizeof (float), .kind = SHARED);
    }
    memcpy (tensor->grad, grad, size * sizeof (float));
}

// COMPARISON OPS
bool
tensor_same_shape (tensor_t *a, tensor_t *b, bool debug)
{
    bool same = is_same_shape (a->shape, b->shape, a->ndim, b->ndim);
    if (debug && !same)
    {
        print_metadata (a->shape, a->ndim);
        printf (" != ");
        print_metadata (b->shape, b->ndim);
        printf ("\n");
    }
    return same;
}

bool
tensor_equals (tensor_t *a, tensor_t *b, bool with_grad)
{
    if (!tensor_same_shape (a, b, false))
        return false;
    if (!is_equal_data (a->data, b->data, a->size))
        return false;
    if (with_grad && a->requires_grad != b->requires_grad)
        return false;
    if (a->requires_grad && b->requires_grad && !is_equal_data (a->grad, b->grad, a->size))
        return false;
    return true;
}

// UNARY OPS
tensor_t *
tensor_neg (tensor_t *a)
{
    return tensor_mul_tf (a, -1.0);
}

tensor_t *
tensor_exp (tensor_t *a)
{
    return tensor_pow_ft (expf (1.0), a);
}

tensor_t *
tensor_relu (tensor_t *a)
{
    tensor_t *out = tensor_init (a->shape, a->ndim, a->requires_grad, op_relu ());
    tensor_add_child (out, a);
    return out;
}

// BINARY OPS
tensor_t *
tensor_add (tensor_t *a, tensor_t *b)
{
    return tensor_add_tt (a, b);
}

tensor_t *
tensor_sub (tensor_t *a, tensor_t *b)
{
    return tensor_sub_tt (a, b);
}

tensor_t *
tensor_mul (tensor_t *a, tensor_t *b)
{
    return tensor_mul_tt (a, b);
}

tensor_t *
tensor_div (tensor_t *a, tensor_t *b)
{
    return tensor_div_tt (a, b);
}

tensor_t *
tensor_pow (tensor_t *a, tensor_t *b)
{
    return tensor_pow_tt (a, b);
}

tensor_t *
tensor_add_tt (tensor_t *a, tensor_t *b)
{
    ASSERT (tensor_same_shape (a, b, true), "Add error :: Shape mismatch");
    tensor_t *out
        = tensor_init (a->shape, a->ndim, a->requires_grad || b->requires_grad, op_add ());
    tensor_add_child (out, a);
    tensor_add_child (out, b);

    return out;
}

tensor_t *
tensor_add_tf (tensor_t *a, float b)
{
    tensor_t       *out = tensor_init (a->shape, a->ndim, a->requires_grad, op_add ());
    smart tensor_t *tmp = tensor ((float[]) { b }, (int[]) { 1 }, 1, false);
    tensor_add_child (out, a);
    tensor_add_child (out, tmp);

    return out;
}

tensor_t *
tensor_add_ft (float a, tensor_t *b)
{
    return tensor_add_tf (b, a);
}

tensor_t *
tensor_sub_tt (tensor_t *a, tensor_t *b)
{
    smart tensor_t *nb = tensor_neg (b);
    return tensor_add (a, nb);
}

tensor_t *
tensor_sub_tf (tensor_t *a, float b)
{
    return tensor_add_tf (a, -b);
}

tensor_t *
tensor_sub_ft (float a, tensor_t *b)
{
    return tensor_add_ft (a, tensor_neg (b));
}

tensor_t *
tensor_mul_tt (tensor_t *a, tensor_t *b)
{
    ASSERT (tensor_same_shape (a, b, true), "Mul error :: Shape mismatch");
    tensor_t *out
        = tensor_init (a->shape, a->ndim, a->requires_grad || b->requires_grad, op_mul ());
    tensor_add_child (out, a);
    tensor_add_child (out, b);

    return out;
}

tensor_t *
tensor_mul_tf (tensor_t *a, float b)
{
    tensor_t       *out = tensor_init (a->shape, a->ndim, a->requires_grad, op_mul ());
    smart tensor_t *tmp = tensor ((float[]) { b }, (int[]) { 1 }, 1, false);
    tensor_add_child (out, a);
    tensor_add_child (out, tmp);

    return out;
}

tensor_t *
tensor_mul_ft (float a, tensor_t *b)
{
    return tensor_mul_tf (b, a);
}

tensor_t *
tensor_div_tt (tensor_t *a, tensor_t *b)
{
    smart tensor_t *invb = tensor_pow_tf (b, -1.0f);
    return tensor_mul (a, invb);
}

tensor_t *
tensor_div_tf (tensor_t *a, float b)
{
    return tensor_mul_tf (a, 1.0 / b);
}

tensor_t *
tensor_div_ft (float a, tensor_t *b)
{
    smart tensor_t *invb = tensor_pow_tf (b, -1.0f);
    return tensor_mul_ft (a, invb);
}

tensor_t *
tensor_pow_tt (tensor_t *a, tensor_t *b)
{
    ASSERT (tensor_same_shape (a, b, true), "Pow error :: Shape mismatch");
    tensor_t *out
        = tensor_init (a->shape, a->ndim, a->requires_grad || b->requires_grad, op_pow ());
    tensor_add_child (out, a);
    tensor_add_child (out, b);

    return out;
}

tensor_t *
tensor_pow_tf (tensor_t *a, float b)
{
    tensor_t       *out = tensor_init (a->shape, a->ndim, a->requires_grad, op_pow ());
    smart tensor_t *tmp = tensor ((float[]) { b }, (int[]) { 1 }, 1, false);
    tensor_add_child (out, a);
    tensor_add_child (out, tmp);

    return out;
}

tensor_t *
tensor_pow_ft (float a, tensor_t *b)
{
    tensor_t       *out = tensor_init (b->shape, b->ndim, b->requires_grad, op_pow ());
    smart tensor_t *tmp = tensor ((float[]) { a }, (int[]) { 1 }, 1, false);
    tensor_add_child (out, tmp);
    tensor_add_child (out, b);

    return out;
}

// REDUCE OPS
tensor_t *
tensor_sum (tensor_t *tensor, const int axes[], size_t n_axis)
{
    if (!n_axis || axes == NULL)
    {
        tensor_t *out = tensor_init ((int[]) { 1 }, 1, tensor->requires_grad, op_sum (NULL));
        tensor_add_child (out, tensor);
        return out;
    }

    tensor_t *reduced = tensor_sum_dim (tensor, axes[0]);
    for (int i = 1; i < n_axis; i++)
    {
        TENSOR_REBIND (reduced, tensor_sum_dim (reduced, axes[i]));
    }

    return reduced;
}

tensor_t *
tensor_sum_dim (tensor_t *tensor, int axis)
{
    axis = (axis < 0) ? (int) tensor->ndim + axis : axis;
    ASSERT (axis >= 0 && axis < (int) tensor->ndim, "Invalid axis %d for ndim %zu", axis,
            tensor->ndim);

    // Target shape
    int shape[(size_t) tensor->ndim];
    memcpy (shape, tensor->shape, tensor->ndim * sizeof (int));
    shape[axis] = 1;

    // Sum reduce
    axis_params_t *params = unique_ptr (axis_params_t, { .axis = axis }, axis_params_destructor);
    tensor_t      *out = tensor_init (shape, tensor->ndim, tensor->requires_grad, op_sum (params));
    tensor_add_child (out, tensor);

    // TODO: Squeeze
    return out;
}

// MOVEMENT OPS
tensor_t *
tensor_clone (tensor_t *tensor)
{
    tensor_t *out = tensor_init (tensor->shape, tensor->ndim, tensor->requires_grad, op_copy ());
    tensor_add_child (out, tensor);
    return out;
}

tensor_t *
tensor_reshape (tensor_t *tensor, int shape[], size_t ndim)
{
    size_t size = get_size (shape, ndim);
    ASSERT (size == tensor->size, "Size mismatch %zu != %zu", size, tensor->size);

    tensor_t *reshaped = tensor_init (shape, ndim, tensor->requires_grad, op_ref ());
    tensor_add_child (reshaped, tensor);

    return reshaped;
}

tensor_t *
tensor_transpose (tensor_t *tensor, int axis1, int axis2)
{
    ASSERT (tensor->ndim + axis1 >= 0 && axis1 < tensor->ndim, "Axis1 out of bounds: got %d",
            axis1);
    ASSERT (tensor->ndim + axis2 >= 0 && axis2 < tensor->ndim, "Axis2 out of bounds: got %d",
            axis2);

    axis1 = (axis1 < 0) ? tensor->ndim + axis1 : axis1;
    axis2 = (axis2 < 0) ? tensor->ndim + axis2 : axis2;

    // set shape
    int shape[tensor->ndim];
    memcpy (shape, tensor->shape, sizeof (shape));
    shape[axis1] = tensor->shape[axis2];
    shape[axis2] = tensor->shape[axis1];

    // set stride
    tensor_t *transposed = tensor_reshape (tensor, shape, tensor->ndim);
    memcpy (transposed->stride, tensor->stride, tensor->ndim * sizeof (int));
    transposed->stride[axis1] = tensor->stride[axis2];
    transposed->stride[axis2] = tensor->stride[axis1];

    return transposed;
}

tensor_t *
tensor_slice (tensor_t *tensor, slice_t range[])
{
    // Compute new shape
    int shape[tensor->ndim];
    normalize_range (range, tensor->shape, tensor->ndim);
    for (int d = 0; d < tensor->ndim; d++)
    {
        shape[d] = abs (range[d].stop - range[d].start) / range[d].step
                   + (abs (range[d].stop - range[d].start) % range[d].step != 0 ? 1 : 0);
    }

    slice_params_t *params = unique_ptr (slice_params_t, { .range = NULL, .ndim = tensor->ndim },
                                         slice_params_destructor);
    params->range          = malloc (sizeof (slice_t) * tensor->ndim);
    memcpy (params->range, range, sizeof (slice_t) * tensor->ndim);

    tensor_t *sliced = tensor_init (shape, tensor->ndim, tensor->requires_grad, op_slice (params));
    tensor_add_child (sliced, tensor);

    return sliced;
}

tensor_t *
tensor_cat (tensor_t *tensors[], size_t n_tensors, int axis)
{
    // Validate inputs
    ASSERT (n_tensors > 0, "tensor_cat requires at least one tensor");

    int ndim = tensors[0]->ndim;
    axis     = (axis < 0) ? ndim + axis : axis;

    ASSERT (axis >= 0 && axis < ndim, "Invalid axis %d for ndim %d", axis, ndim);

    // Check shape compatibility
    for (size_t i = 1; i < n_tensors; ++i)
    {
        ASSERT (tensors[i]->ndim == ndim, "Shape mismatch: tensor %zu has ndim %d, expected %d", i,
                tensors[i]->ndim, ndim);
        for (int d = 0; d < ndim; ++d)
        {
            if (d != axis)
            {
                ASSERT (tensors[i]->shape[d] == tensors[0]->shape[d],
                        "Shape mismatch along dim %d: %d vs %d", d, tensors[i]->shape[d],
                        tensors[0]->shape[d]);
            }
        }
    }

    // Compute new shape
    int shape[ndim];
    memcpy (shape, tensors[0]->shape, sizeof (shape));
    for (size_t i = 1; i < n_tensors; ++i)
    {
        shape[axis] += tensors[i]->shape[axis];
    }

    // Determine requires_grad
    bool requires_grad = false;
    for (size_t i = 0; i < n_tensors; ++i)
    {
        if (tensors[i]->requires_grad)
        {
            requires_grad = true;
            break;
        }
    }

    // Create output tensor
    axis_params_t *params = unique_ptr (axis_params_t, { .axis = axis }, axis_params_destructor);
    tensor_t      *cated  = tensor_init (shape, ndim, requires_grad, op_cat (params));

    for (size_t i = 0; i < n_tensors; i++)
    {
        tensor_add_child (cated, tensors[i]);
    }
    return cated;
}

// FORCING OPS
void
tensor_forward (tensor_t *tensor)
{
    if (tensor->op == NULL)
    {
        return;
    }

    for (size_t i = 0; i < tensor->n_children; ++i)
    {
        tensor_forward (tensor->children[i]);
    }
    forward (tensor);
}

void
tensor_backward (tensor_t *tensor)
{
    if (tensor->op == NULL)
    {
        log_error ("Cannot perform backward on tensor that has no op.\n");
        return;
    }
    if (!tensor->requires_grad)
    {
        log_error ("Cannot perform backward on tensor that has no grad.\n");
        return;
    }
    if (tensor->size != 1)
    {
        log_error ("Backward operation only supported for scalar tensors.\n");
        return;
    }
    if (tensor->children == NULL)
    {
        log_warn ("Node %p has no children.", (void *) tensor);
        return;
    }
    tensor_forward (tensor);
    if (tensor->grad == NULL)
    {
        tensor->grad = smalloc (.nmemb = tensor->size, .size = sizeof (float), .kind = SHARED);
        tensor_init_grad (tensor);
    }

    int        capacity = 16;
    int        count    = 0;
    tensor_t **nodes    = malloc (capacity * sizeof (tensor_t *));

    build_topo (tensor, &nodes, &count, &capacity);

    for (int i = count - 1; i >= 0; --i)
    {
        backward (nodes[i]);
    }

    free (nodes);
}

void
tensor_print (tensor_t *tensor, flag_t flags)
{
    tensor_forward (tensor);

    printf ("Tensor @ %p\n", (void *) tensor);
    printf ("Shape:\t");
    print_metadata (tensor->shape, tensor->ndim);
    printf ("\n");

    if (flags & PRINT_STRIDE)
    {
        printf ("Stride:\t");
        print_metadata (tensor->stride, tensor->ndim);
        printf ("\n");
    }

    if (flags & PRINT_CHILDREN)
    {
        if (!tensor->children)
        {
            printf ("Children: NULL\n");
        }
        else
        {
            printf ("Children:\n");
            for (int i = 0; i < tensor->n_children; ++i)
            {
                printf ("\tChild %d: Tensor @ %p\n", i + 1, (void *) tensor->children[i]);
            }
        }
    }

    if (flags & PRINT_DATA)
    {
        if (tensor->data)
        {
            printf ("Data @ %p:\n", (void *) tensor->data);
            print_data (tensor->data, tensor->shape, tensor->stride, tensor->ndim);
        }
        else
        {
            printf ("Data: NULL\n");
        }
    }

    if (flags & PRINT_GRAD)
    {
        if (tensor->grad)
        {
            printf ("Grad @ %p:\n", (void *) tensor->grad);
            print_data (tensor->grad, tensor->shape, tensor->stride, tensor->ndim);
        }
        else
        {
            printf ("Grad: NULL\n");
        }
    }
    printf ("\n");
}