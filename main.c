#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor_rand((int[]){5, 10}, 2, false);
    tensor_t *b = tensor_slice(a, (slice_t[]){(slice_t){0, 5, 1}, (slice_t){0, -1, 1}});
    tensor_print(a, PRINT_ALL);
    tensor_print(b, PRINT_ALL);

    tensor_free(a, true);
    tensor_free(b, true);
    return 0;
}