#include <stdio.h>
#include "tensor.h"
#include "ops.h"
#include "logger.h"

int log_level = LOG_LEVEL_DEBUG;

int main() {
    tensor_t* t1 = tensor((float[]){2., 4., 6.}, (int[]){3}, 1, true);
    tensor_t* t2 = tensor((float[]){1., 2., 0.}, (int[]){3}, 1, true);

    printf("t1 \n");
    tensor_print(t1);
    printf("\n");

    printf("t2 \n");
    tensor_print(t2);
    printf("\n");

    tensor_t* y = tensor_div(t1, t2);
    printf("y \n");
    tensor_print(y);
    printf("\n");

    tensor_t* loss = tensor_sum(y);
    tensor_backward(loss);

    printf("y backward\n");
    tensor_print(y);
    printf("\n");

    printf("t1 backward\n");
    tensor_print(t1);
    printf("\n");

    printf("t2 backward\n");
    tensor_print(t2);
    printf("\n");

    // recursively free all tensors in the graph
    tensor_free(loss, true); 
    return 0;
}