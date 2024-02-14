#ifndef ITERATOR_H
#define ITERATOR_H

#include "tensor.h"
#include "slice.h"
#include "helpers.h"
#include <stdbool.h>

typedef struct
{
    tensor_t *tensor;
    slice_t *range;
    int *indices;
    int count;
    int dim;
} iterator_t;

iterator_t iterator(tensor_t *tensor, slice_t *range);
bool iterator_has_next(iterator_t *iterator);
float iterator_next(iterator_t *iterator);

#endif