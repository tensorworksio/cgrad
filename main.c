#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor_rand((int[]){3, 3}, 2, false);
    tensor_t *b = tensor_slice(a, (slice_t[]){SLICE_ALL, SLICE_ONE(0)}, 2);
    tensor_print(a);
    tensor_print(b);

    return 0;
}