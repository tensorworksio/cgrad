#include "tensor.h"
#include "ops.h"
#include "log.h"

int main() {
    log_set_level(LOG_INFO);

    tensor_t* a = tensor_rand((int[]){3, 3}, 2, false);
    tensor_t* b = tensor_slice(a, (slice_t[]){(slice_t){0, 3, 2}, (slice_t){0, 3, 2}}, 2);
    tensor_print(a);
    tensor_print(b);

    return 0;
}