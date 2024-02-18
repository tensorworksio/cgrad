#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor_rand((int[]){10, 10}, 2, true);
    tensor_t *b = tensor_rand((int[]){10, 10}, 2, true);

    tensor_t *c = tensor_mul(a, b);
    tensor_t *d = tensor_slice(c, (slice_t[]){{0, 5, 1}, {0, 5, 1}});
    tensor_t *e = tensor_sum(d);

    tensor_backward(e);

    tensor_print(a, PRINT_ALL);
    tensor_print(b, PRINT_ALL);
    tensor_print(c, PRINT_ALL);
    tensor_print(d, PRINT_ALL);
    tensor_print(e, PRINT_ALL);

    tensor_free(e, true);
}