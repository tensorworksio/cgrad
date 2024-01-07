#include <stdio.h>
#include "tensor.h"
#include "ops.h"
#include "log.h"

int main() {
    log_set_level(LOG_INFO);

    tensor_t* a = tensor((float[]){1., -2., 3., -4.}, (int[]){4}, 1, true);
    tensor_t* b = tensor_reshape(a, (int[]){2, 2}, 2);

    tensor_t* y = tensor_sum(b);
    tensor_backward(y);

    tensor_print(y);
    tensor_print(b);
    tensor_print(a);

    tensor_free(y, true);
    return 0;
}