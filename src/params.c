#include "params.h"

void
slice_params_destructor (void *ptr, void *meta)
{
    slice_params_t *params = (slice_params_t *) ptr;
    free (params->range);
}
