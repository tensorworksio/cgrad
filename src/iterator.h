#ifndef ITERATOR_H
#define ITERATOR_H

#include "tensor.h"
#include "helpers.h"
#include <stdbool.h>

typedef struct
{
    int ndim;
    slice_t *range;
    int *stride;
    int *indices;
    int count;
    int size;

} iterator_t;

iterator_t iterator(slice_t *range, int *stride, int ndim);
iterator_t iterator_tensor(tensor_t *tensor);
void iterator_reset(iterator_t *iterator);
void iterator_free(iterator_t *iterator);
bool iterator_has_next(iterator_t *iterator);
int iterator_next(iterator_t *iterator);
int iterator_size(iterator_t *iterator);
int iterator_eod(iterator_t *it);
int iterator_sod(iterator_t *it);

#endif