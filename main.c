#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor_rand((int[]){3, 4}, 2, false);
    tensor_print(a);
    tensor_t *b = tensor_transpose(a, 0, 1);
    tensor_print(b);
    return 0;
}