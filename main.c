#include <stdio.h>
#include "tensor.h"
#include "ops.h"

int main() {
    tensor_t* x1 = tensor_create_random(10);
    tensor_t* x2 = tensor_create_random(10);

    printf("x1 \n");
    tensor_print(x1);
    printf("\n");

    printf("x2 \n");
    tensor_print(x2);
    printf("\n");

    tensor_t* y = mul(x1, x2);
    printf("y \n");
    tensor_print(y);
    printf("\n");

    tensor_t* loss = sum(y);
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

    tensor_free(y);
    tensor_free(x1);
    tensor_free(x2);

    return 0;
}