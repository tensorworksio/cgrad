#include "forward.h"
#include "log.h"
#include "tensor.h"

int
main ()
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 2., 4., 6. }, (int[]) { 3 }, 1, true);
    smart tensor_t *b = tensor ((float[]) { 1., 2., 0. }, (int[]) { 3 }, 1, true);
    // c = a + b
    smart tensor_t *c = tensor_add (a, b);
    // c = c - 1
    TENSOR_REBIND (c, tensor_sub_tf (c, 1.0f));
    // d = c ** 3
    smart tensor_t *d = tensor_pow_tf (c, 3.);
    // e = relu(d)
    smart tensor_t *e = tensor_relu (d);
    // f = sum(e)
    smart tensor_t *f = tensor_sum (e, 0);

    tensor_backward (f); // compute gradients

    tensor_print (a, PRINT_ALL); // print tensors a.data and a.grad = d(f)/d(a)
    tensor_print (b, PRINT_ALL); // print tensors b.data and b.grad = d(f)/d(b)

    return 0;
}