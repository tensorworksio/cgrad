#ifndef ITERATOR_H
#define ITERATOR_H

#include "tensor.h"
#include "slice.h"
#include "helpers.h"
#include <stdbool.h>

typedef struct
{
    tensor_t *tensor;
    int *indices;
    int count;
    int dim;
} iterator_t;

iterator_t *iterator_create(tensor_t *tensor);
bool iterator_has_next(iterator_t *iterator);
float iterator_next(iterator_t *iterator);
void iterator_free(iterator_t *iterator);

#endif