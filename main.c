#include "tensor.h"
#include "ops.h"
#include "log.h"
#include "iterator.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor_rand((int[]){5, 10}, 2, false);
    tensor_t *b = tensor_rand((int[]){5, 10}, 2, false);

    tensor_t *c = tensor_add(a, b);
    tensor_forward(c);

    tensor_print(a, PRINT_ALL);
    tensor_print(b, PRINT_ALL);
    tensor_print(c, PRINT_ALL);

    tensor_free(c, true);

    return 0;
}