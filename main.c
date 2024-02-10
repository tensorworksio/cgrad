#include "tensor.h"
#include "ops.h"
#include "log.h"
#include "iterator.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., -2., 3., -4.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_sum(a);
    tensor_print(res, PRINT_ALL);

    tensor_t *expected = tensor((float[]){-2.}, (int[]){1}, 1, false);
    ASSERT(tensor_equals(res, expected, true), "sum_t failed");

    tensor_free(res, true);
    tensor_free(expected, true);

    return 0;
}