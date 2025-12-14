#include "forward.h"
#include "log.h"
#include "tensor.h"

int
main ()
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4., 5., 6. }, (int[]) { 2, 3 }, 2, true);
    smart tensor_t *f = tensor_sum (a, 2, (int[]) { 0, 1 });

    tensor_backward (f); // compute gradients

    tensor_print (a, PRINT_ALL); // print tensors a.data and a.grad = d(f)/d(a)
    tensor_print (f, PRINT_ALL); // print tensors b.data and b.grad = d(f)/d(b)

    return 0;
}