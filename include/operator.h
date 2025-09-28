#ifndef OPERATOR_H
#define OPERATOR_H

#include "csptr/smart_ptr.h"
#include "slice.h"
#include <stdbool.h>
#include <stdlib.h>

// Forward declaration tensor_t
typedef struct tensor tensor_t;

typedef struct op
{
    const char *name;                  // For identification/debugging
    void       *params;                // Operation-specific parameters
    void (*backward) (tensor_t *self); // Backward function
    void (*forward) (tensor_t *self);  // Forward function
} op_t;

static inline op_t
op_init (const char *name, void (*forward) (tensor_t *), void (*backward) (tensor_t *),
         void       *params, void (*destructor) (void *))
{
    return (op_t) { .name = name, .params = params, .backward = backward, .forward = forward };
};

static inline void
op_destructor (void *ptr, void *meta)
{
    op_t *op = (op_t *) ptr;
    if (op->params)
    {
        sfree (op->params);
    }
    (void) meta;
};

// Operator params

typedef struct slice_params
{
    slice_t *range;
    int      ndim; // Added for convenience if needed
} slice_params_t;

typedef struct cat_params
{
    int axis;
} cat_params_t;

static void
slice_params_destructor (void *ptr, void *meta)
{
    slice_params_t *params = (slice_params_t *) ptr;
    free (params->range);
}

static void
cat_params_destructor (void *ptr, void *meta)
{
    // cat_params_t has no dynamic fields, so no-op (or add if needed)
    (void) ptr;
    (void) meta;
}

// Operator definiton

#include "backward.h"
#include "forward.h"

static inline op_t *
op_add (void)
{
    return unique_ptr (
        op_t, { .name = "add", .params = NULL, .backward = backward_add, .forward = forward_add },
        op_destructor);
}

static inline op_t *
op_mul (void)
{
    return unique_ptr (
        op_t, { .name = "mul", .params = NULL, .backward = backward_mul, .forward = forward_mul },
        op_destructor);
}

static inline op_t *
op_pow (void)
{
    return unique_ptr (
        op_t, { .name = "pow", .params = NULL, .backward = backward_pow, .forward = forward_pow },
        op_destructor);
}

static inline op_t *
op_sum (void)
{
    return unique_ptr (
        op_t, { .name = "sum", .params = NULL, .backward = backward_sum, .forward = forward_sum },
        op_destructor);
}

static inline op_t *
op_relu (void)
{
    return unique_ptr (
        op_t,
        { .name = "relu", .params = NULL, .backward = backward_relu, .forward = forward_relu },
        op_destructor);
}

static inline op_t *
op_copy (void)
{
    return unique_ptr (
        op_t,
        { .name = "copy", .params = NULL, .backward = backward_copy, .forward = forward_copy },
        op_destructor);
}

static inline op_t *
op_ref (void)
{
    return unique_ptr (
        op_t, { .name = "ref", .params = NULL, .backward = backward_ref, .forward = forward_ref },
        op_destructor);
}

static inline op_t *
op_slice (slice_t *range, int ndim)
{
    slice_params_t *params
        = unique_ptr (slice_params_t, { .range = NULL, .ndim = ndim }, slice_params_destructor);
    params->range = malloc (sizeof (slice_t) * ndim);
    memcpy (params->range, range,
            sizeof (slice_t) * ndim); // Copy range to avoid external management

    return unique_ptr (
        op_t,
        { .name = "slice", .params = params, .backward = backward_slice, .forward = forward_slice },
        op_destructor);
}

static inline op_t *
op_cat (int axis)
{
    cat_params_t *params = unique_ptr (cat_params_t, { .axis = axis }, cat_params_destructor);

    return unique_ptr (
        op_t, { .name = "cat", .params = params, .backward = backward_cat, .forward = forward_cat },
        op_destructor);
}

#endif // OPERATOR_H