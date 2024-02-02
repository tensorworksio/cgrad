#ifndef SLICE_H
#define SLICE_H

#include <stdlib.h>

#define SLICE_ALL \
    (slice_t) { 0, -1, 1 }
#define SLICE_ONE(start) \
    (slice_t) { (start), (start) + 1, 1 }
#define SLICE_RANGE(start, stop) \
    (slice_t) { (start), (stop), 1 }

typedef struct
{
    int start;
    int stop;
    int step;
} slice_t;

#endif