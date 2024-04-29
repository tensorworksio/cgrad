#include "tensor.h"
#include "ops.h"
#include "log.h"
#include "time.h"

int main()
{
    log_set_level(LOG_INFO);
    tensor_t *a = tensor((float[]){1., 2., 3., 4., 5., 6., 7., 8.}, (int[]){4, 2}, 2, false);
    tensor_t *b = tensor_slice(a, (slice_t[]){SLICE_ONE(3), SLICE_ALL});

    tensor_forward(b);
    tensor_print(a, PRINT_ALL);
    tensor_print(b, PRINT_ALL);

    return 0;
}