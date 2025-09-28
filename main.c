#include "forward.h"
#include "log.h"
#include "tensor.h"

int
main ()
{
    log_set_level (LOG_INFO);

    smart tensor_t *a
        = tensor ((float[]) { 1., 2., 3., 4., 5., 6., 7., 8. }, (int[]) { 4, 2 }, 2, true);

    smart tensor_t *b
        = tensor ((float[]) { 1., 2., 3., 4., 5., 6., 7., 8. }, (int[]) { 4, 2 }, 2, true);

    smart tensor_t *c = tensor_cat ((tensor_t *[]) { a, b }, 2, 0);
    smart tensor_t *f = tensor_sum (c);

    tensor_backward (f);

    tensor_print (a, PRINT_ALL);
    tensor_print (b, PRINT_ALL);
    tensor_print (c, PRINT_ALL);
    tensor_print (f, PRINT_ALL);

    return 0;
}