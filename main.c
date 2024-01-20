#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor_rand((int[]){2, 3, 5, 10}, 4, false);
    tensor_print(a);
    tensor_t *b = tensor_transpose(a, 2, 3);
    tensor_print(b);
    return 0;
}