#include "iterator.h"
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

    tensor_forward (b);
    tensor_print (a, PRINT_ALL);
    tensor_print (b, PRINT_ALL);

    smart iterator_t *it = iterator (b->range, a->stride, b->ndim);

    while (iterator_has_next (it))
    {
        printf ("%f, ", a->data[iterator_next (it)]);
    }

    return 0;
}