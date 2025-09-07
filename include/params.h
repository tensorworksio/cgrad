#ifndef PARAMS_H
#define PARAMS_H

#include "slice.h"

// Operation parameter structures
typedef struct
{
    slice_t *range;
} slice_params_t;

typedef struct
{
    int axis;
} cat_params_t;

// Parameter destructors
void slice_params_destructor (void *ptr, void *meta);

#endif // PARAMS_H