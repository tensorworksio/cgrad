#include "tensor.h"
#include "ops.h"
#include "log.h"
#include "time.h"

int main()
{
    log_set_level(LOG_INFO);
    tensor_t *a = tensor_rand((int[]){2, 3}, 2, false);
    tensor_t *b = tensor_rand((int[]){2, 3}, 2, false);
    tensor_t *c = tensor_rand((int[]){2, 3}, 2, false);

    tensor_t *d = tensor_add(a, b);
    tensor_t *e = tensor_add(b, c);
    tensor_t *f = tensor_add(d, e);

    tensor_free(f, true);
    return 0;
}