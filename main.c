#include <stdio.h>
#include "tensor.h"
#include "ops.h"
#include "log.h"

int main() {
    log_set_level(LOG_INFO);

    tensor_t* a = tensor((float[]){1., -2., 3., -4.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_sum(a);
    tensor_t* expected = tensor((float[]){-2.}, (int[]){1}, 1, false);

    tensor_print(a);
    tensor_print(res);
    tensor_print(expected);

    if (!tensor_equals(res, expected, true)) {
        printf("tensor_equals failed\n");
    }
    tensor_free(res, true);
    tensor_free(expected, true);
    return 0;
}