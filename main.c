#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){2., 4., 6.}, (int[]){3}, 1, true);
    tensor_t *b = tensor((float[]){1., 2., 0.}, (int[]){3}, 1, true);
    // c = a + b
    tensor_t *c = tensor_add(a, b);
    // c = c - 1
    c = tensor_sub_tf(c, 1.);
    // d = c ** 3
    tensor_t *d = tensor_pow_tf(c, 3.);
    // e = relu(d)
    tensor_t *e = tensor_relu(d);
    // f = sum(e)
    tensor_t *f = tensor_sum(e);

    tensor_backward(f); // compute gradients

    tensor_print(a, PRINT_ALL); // print tensors a.data and a.grad = d(f)/d(a)
    tensor_print(b, PRINT_ALL); // print tensors b.data and b.grad = d(f)/d(b)

    // recursively free all tensors in the graph
    tensor_free(f, true);
    return 0;
}