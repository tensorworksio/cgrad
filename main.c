#include "iterator.h"
#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor_rand((int[]){3, 5, 10}, 3, false);

    tensor_print(a, PRINT_ALL);

    iterator_t *it = iterator_create(a);
    float next;
    while (iterator_has_next(it))
    {
        next = iterator_next(it);
        printf("%f\n", next);
    }

    tensor_free(a, true);
    iterator_free(it);
}