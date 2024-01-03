#include <stdio.h>
#include "tensor.h"
#include "ops.h"
#include "logger.h"

int log_level = LOG_LEVEL_DEBUG;

int main() {
    tensor_t* t = tensor((float[]){1., 2., 3.}, (int[]){3}, 1, true);

    printf("t \n");
    tensor_print(t);
    printf("\n");

    tensor_t* y = tensor_exp(t);
    printf("y \n");
    tensor_print(y);
    printf("\n");

    tensor_t* loss = tensor_sum(y);
    tensor_backward(loss);

    printf("y backward\n");
    tensor_print(y);
    printf("\n");

    printf("t backward\n");
    tensor_print(t);
    printf("\n");

    // recursively free all tensors in the graph
    tensor_free(loss, true); 
    return 0;
}