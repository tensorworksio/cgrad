#include "log.h"
#include "ops.h"
#include "tensor.h"

int
main ()
{
    log_set_level (LOG_INFO);

    smart tensor_t *a
        = tensor ((float[]) { 1., 2., 3., 4., 5., 6., 7., 8. }, (int[]) { 2, 2, 2 }, 3, true);
    smart tensor_t *b = tensor_clone (a);
    smart tensor_t *c = tensor_sum (b);

    tensor_backward (c); // compute gradients

    tensor_print (a, PRINT_ALL); // print tensors a.data and a.grad = d(f)/d(a)
    tensor_print (b, PRINT_ALL); // print tensors b.data and b.grad = d(f)/d(b)

    return 0;
}