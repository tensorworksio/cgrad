#ifndef ITERATOR_H
#define ITERATOR_H

#include "slice.h"
#include "assert.h"
#include <stdbool.h>

// typedef struct
// {
//     int (*next)(void *);
//     bool (*has_next)(void *);
//     void (*reset)(void *);
//     void (*free)(void *);
//     int (*size)(void *);
// } iterator_t;

typedef struct
{
    int ndim;
    slice_t *range;
    int *shape;
    int *stride;
    int *indices;

    bool has_next;

} iterator_t;

iterator_t iterator(slice_t *range, int *stride, int ndim);

void iterator_reset(iterator_t *it);
void iterator_free(iterator_t *it);
void iterator_update(iterator_t *it);

bool iterator_has_next(iterator_t *it);
int iterator_index(iterator_t *it);
int iterator_next(iterator_t *it);

int iterator_size(iterator_t *it);
int iterator_eod(iterator_t *it);
int iterator_sod(iterator_t *it);

#endif