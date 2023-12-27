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
    array_t* array = array_create_random(10);
    printf("array \n");
    array_print(array);
    printf("\n");

    array_t* array2 = add(array, 1.);
    printf("array2 \n");
    array_print(array2);
    printf("\n");

    array_t* array3 = add(array2, 2.);
    printf("array3 \n");
    array_print(array3);
    printf("\n");

    set_grad(array3, 1.);
    printf("array3 grad set \n");
    array_print(array3);
    printf("\n");

    array3->backward(array3);

    printf("array2 backward\n");
    array_print(array2);
    printf("\n");

    printf("array backward\n");
    array_print(array);
    printf("\n");

    array_free(array3);
    return 0;
}