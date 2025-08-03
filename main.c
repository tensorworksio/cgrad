#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    smart tensor_t *a = tensor((float[]){1., 2., 3., 4., 5., 6.}, (int[]){3, 2}, 2, true);
    smart tensor_t *b = tensor_mul_ft(2.f, a);
    smart tensor_t *c = tensor_mul_ft(3.f, a);
    smart tensor_t *d = tensor_add(b, c);
    smart tensor_t *e = tensor_sum(d);

    tensor_backward(e); // compute gradients

    tensor_print(a, PRINT_ALL); // print tensors a.data and a.grad = d(f)/d(a)
    tensor_print(b, PRINT_ALL); // print tensors b.data and b.grad = d(f)/d(b)
    tensor_print(c, PRINT_ALL);
    tensor_print(d, PRINT_ALL);
    tensor_print(e, PRINT_ALL);

    return 0;
}