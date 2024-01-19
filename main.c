#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor_rand((int[]){3, 5, 10}, 3, false);
    tensor_print(a);
    tensor_t *b = tensor_transpose(a, 1, 2);
    tensor_print(b);
    return 0;
}