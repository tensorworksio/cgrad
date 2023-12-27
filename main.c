#include <stdio.h>
#include "array.h"
#include "ops.h"

void set_grad(array_t* self, float grad) {
    for (int i = 0; i < self->size; i++)
    {
        self->grad[i] = grad;
    }
}

int main() {
    array_t* x1 = array_create_random(10);
    array_t* x2 = array_create_random(10);

    printf("x1 \n");
    array_print(x1);
    printf("\n");

    printf("x2 \n");
    array_print(x2);
    printf("\n");

    array_t* y = mul(x1, x2);
    printf("y \n");
    array_print(y);
    printf("\n");

    set_grad(y, 1.0);
    printf("y grad set\n");
    array_print(y);
    printf("\n");

    backward(y);

    printf("x1 backward\n");
    array_print(x1);
    printf("\n");

    printf("x2 backward\n");
    array_print(x2);
    printf("\n");

    array_free(y);
    array_free(x1);
    array_free(x2);

    return 0;
}