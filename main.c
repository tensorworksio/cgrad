#include "forward.h"
#include "log.h"
#include "tensor.h"

int
main ()
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4., 5., 6. }, (int[]) { 2, 3 }, 2, true);
    smart tensor_t *b = tensor ((float[]) { 1., 2., 3., 4., 5., 6. }, (int[]) { 3, 2 }, 2, true);
    smart tensor_t *c = tensor_matmul (a, b);
    smart tensor_t *f = tensor_sum (c, NULL, 0);
    tensor_backward (f); // compute gradients

    tensor_print (a, PRINT_ALL); // print tensors a.data and a.grad = d(f)/d(a)
    tensor_print (b, PRINT_ALL); // print tensors b.data and b.grad = d(f)/d(b)
    tensor_print (c, PRINT_ALL); // print tensors c.data and c.grad = d(f)/d(c)
    tensor_print (f, PRINT_ALL); // print tensors f.data and f.grad = d(f)/d(f)

    return 0;
}