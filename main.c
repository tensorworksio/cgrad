#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor_rand((int[]){5, 10}, 2, false);
    tensor_t *b = tensor_rand((int[]){5, 10}, 2, false);
    tensor_t *c = tensor_add_tt(a, b);
    tensor_print(a, PRINT_ALL);
    tensor_print(b, PRINT_ALL);
    tensor_print(c, PRINT_ALL);
    
    return 0;
}