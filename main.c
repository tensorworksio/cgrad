#include "log.h"
#include "ops.h"
#include "tensor.h"

int
main ()
{
    log_set_level (LOG_INFO);

    smart tensor_t *a
        = tensor ((float[]) { 1., 2., 3., 4., 5., 6., 7., 8. }, (int[]) { 4, 2 }, 2, true);
    smart tensor_t *b = tensor_slice (a, (slice_t[]) { (slice_t) { 1, 3, 1 }, SLICE_ALL });
    smart tensor_t *f = tensor_sum (b);

    tensor_backward (f);

    tensor_print (a, PRINT_ALL);
    tensor_print (b, PRINT_ALL);
    tensor_print (f, PRINT_ALL);

    return 0;
}