#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor_rand((int[]){3, 5, 10}, 3, false);
    tensor_t *b = tensor_rand((int[]){3, 3, 10}, 3, false);
    tensor_t *c = tensor_rand((int[]){3, 2, 10}, 3, false);
    tensor_t *d = tensor_cat((tensor_t *[]){a, b, c}, 3, 1);

    tensor_print(a, PRINT_ALL);
    tensor_print(b, PRINT_ALL);
    tensor_print(c, PRINT_ALL);
    tensor_print(d, PRINT_ALL);

    tensor_free(d, true);
}