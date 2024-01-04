#include <stdio.h>
#include "tensor.h"
#include "ops.h"
#include "log.h"

int main() {
    log_set_level(LOG_INFO);

    tensor_t* t1 = tensor((float[]){2., 4., 6.}, (int[]){3}, 1, true);
    tensor_t* t2 = tensor((float[]){1., 2., 0.}, (int[]){3}, 1, true);

    printf("t1\n");
    tensor_print(t1);

    printf("t2\n");
    tensor_print(t2);

    tensor_t* y = tensor_div(t1, t2);
    printf("y\n");
    tensor_print(y);

    tensor_t* loss = tensor_sum(y);
    tensor_backward(loss);

    printf("y backward\n");
    tensor_print(y);

    printf("t1 backward\n");
    tensor_print(t1);

    printf("t2 backward\n");
    tensor_print(t2);

    // recursively free all tensors in the graph
    tensor_free(loss, true); 
    return 0;
}