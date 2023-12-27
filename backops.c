#include "backops.h"

void backward_add(array_t* self)
{   
    if (self->child == NULL) {
        printf("self->child is NULL\n");
        return;
    }
    for (int i = 0; i < self->size; i++)
    {   
        self->child->grad[i] += self->grad[i];
    }
    if (self->child->backward == NULL) {
        printf("self->child->backward is NULL\n");
        return;
    }
    self->child->backward(self->child);
}