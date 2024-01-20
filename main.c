#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor_rand((int[]){5, 10}, 2, true);
    tensor_t *b = tensor_transpose(a, 0, 1);
    tensor_t *c = tensor_sum(a);
    tensor_backward(c);

    tensor_print(a, PRINT_DATA | PRINT_GRAD);
    tensor_print(b, PRINT_DATA | PRINT_GRAD);
    tensor_print(c, PRINT_DATA | PRINT_GRAD);
    
    return 0;
}