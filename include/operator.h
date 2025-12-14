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
    bool visited;                      // Visited flag to ensure single backward traversal
} op_t;

static inline op_t
op_init (const char *name, void (*forward) (tensor_t *), void (*backward) (tensor_t *),
         void       *params, void (*destructor) (void *))
{
    return (op_t) {
        .name = name, .params = params, .backward = backward, .forward = forward, .visited = false
    };
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
    int      ndim;
} slice_params_t;

typedef struct axis_params
{
    int axis;
} axis_params_t;

static inline void
slice_params_destructor (void *ptr, void *meta)
{
    slice_params_t *params = (slice_params_t *) ptr;
    free (params->range);
}

static inline void
axis_params_destructor (void *ptr, void *meta)
{
    // axis_params_t has no dynamic fields, so no-op (or add if needed)
    (void) ptr;
    (void) meta;
}

// Operator definiton

#include "backward.h"
#include "forward.h"

static inline op_t *
op_add (void)
{
    return unique_ptr (op_t,
                       { .name     = "add",
                         .params   = NULL,
                         .backward = backward_add,
                         .forward  = forward_add,
                         .visited  = false },
                       op_destructor);
}

static inline op_t *
op_mul (void)
{
    return unique_ptr (op_t,
                       { .name     = "mul",
                         .params   = NULL,
                         .backward = backward_mul,
                         .forward  = forward_mul,
                         .visited  = false },
                       op_destructor);
}

static inline op_t *
op_pow (void)
{
    return unique_ptr (op_t,
                       { .name     = "pow",
                         .params   = NULL,
                         .backward = backward_pow,
                         .forward  = forward_pow,
                         .visited  = false },
                       op_destructor);
}

static inline op_t *
op_sum (axis_params_t *params)
{
    return unique_ptr (op_t,
                       { .name     = "sum",
                         .params   = params,
                         .backward = backward_sum,
                         .forward  = forward_sum,
                         .visited  = false },
                       op_destructor);
}

static inline op_t *
op_relu (void)
{
    return unique_ptr (op_t,
                       { .name     = "relu",
                         .params   = NULL,
                         .backward = backward_relu,
                         .forward  = forward_relu,
                         .visited  = false },
                       op_destructor);
}

static inline op_t *
op_copy (void)
{
    return unique_ptr (op_t,
                       { .name     = "copy",
                         .params   = NULL,
                         .backward = backward_copy,
                         .forward  = forward_copy,
                         .visited  = false },
                       op_destructor);
}

static inline op_t *
op_ref (void)
{
    return unique_ptr (op_t,
                       { .name     = "ref",
                         .params   = NULL,
                         .backward = backward_ref,
                         .forward  = forward_ref,
                         .visited  = false },
                       op_destructor);
}

static inline op_t *
op_slice (slice_params_t *params)
{
    return unique_ptr (op_t,
                       { .name     = "slice",
                         .params   = params,
                         .backward = backward_slice,
                         .forward  = forward_slice,
                         .visited  = false },
                       op_destructor);
}

static inline op_t *
op_cat (axis_params_t *params)
{
    return unique_ptr (op_t,
                       { .name     = "cat",
                         .params   = params,
                         .backward = backward_cat,
                         .forward  = forward_cat,
                         .visited  = false },
                       op_destructor);
}
#endif // OPERATOR_H