#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor_rand((int[]){3, 5, 10}, 3, false);
    tensor_t *b = tensor_slice(a, (slice_t[]){SLICE_ALL, SLICE_ALL, (slice_t){0, 5, 1}});

    tensor_print(a, PRINT_ALL);
    tensor_print(b, PRINT_ALL);

    tensor_free(b, true);
}