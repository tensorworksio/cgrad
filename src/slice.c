#include "slice.h"

int slice_size(slice_t range)
{
    return (range.stop - range.start + range.step - 1) / range.step;
}