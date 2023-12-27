#include "array.h"
#include "backops.h"
#include "ops.h"

array_t* add(array_t* a, float b)
{
    array_t* out = array_create(a->size);
    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = a->data[i] + b;
    }
    out->child = a;
    out->backward = backward_add;
    return out;
}
