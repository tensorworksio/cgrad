#include "tensor.h"
#include "ops.h"
#include "log.h"
#include "time.h"

int main()
{
    log_set_level(LOG_INFO);
    clock_t start, end;
    double cpu_time_used;

    tensor_t *a = tensor_rand((int[]){3, 200, 200}, 3, false);

    start = clock();
    iterator_t it = tensor_iterator(a);
    while (iterator_has_next(&it))
    {
        int index = iterator_next(&it);
        a->data[index] = a->data[index] * 2;
    }
    iterator_free(&it);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken by Method 1: %f\n", cpu_time_used);

    start = clock();
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = a->data[i] * 2;
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken by Method 2: %f\n", cpu_time_used);
    return 0;
}