#include <stdio.h>
#include "tensor.h"
#include "ops.h"
#include "logger.h"

int log_level = LOG_LEVEL_DEBUG;

int main() {
    tensor_t* x1 = tensor_create_random((int[]){5, 10}, 2, true); 
    tensor_t* x2 = tensor_create_random((int[]){5, 10}, 2, true);

    printf("x1 \n");
    tensor_print(x1);
    printf("\n");

    printf("x2 \n");
    tensor_print(x2);
    printf("\n");

    // FIXME: this example should output
    // x1.grad = [3 3 3 3 3 3 3 3 3 3]
    tensor_t* y = tensor_mul_ft(3, x2);
    printf("y \n");
    tensor_print(y);
    printf("\n");

    tensor_t* loss = tensor_sum(y);
    backward(loss);

    printf("y backward\n");
    tensor_print(y);
    printf("\n");

    printf("x1 backward\n");
    tensor_print(x1);
    printf("\n");

    printf("x2 backward\n");
    tensor_print(x2);
    printf("\n");

    // recursively free all tensors in the graph
    tensor_free(loss, true); 
    return 0;
}