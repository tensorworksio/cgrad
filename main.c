#include "time.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

int get_size(int shape[], int ndim)
{
    int size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size *= shape[i];
    }
    return size;
}

typedef struct tensor
{
    float *data;
    float *grad;
    bool requires_grad;

    int size;
    int ndim;
    int *shape;
    int *stride;

    struct forward_op *forward_op;
    struct backward_op *backward_op;

} tensor_t;

typedef struct forward_op
{
    tensor_t **sources;
    const void *params;

} forward_op_t;

typedef struct backward_op
{
    tensor_t **targets;
    void *params;

} backward_op_t;

tensor_t *tensor_create(int shape[], int ndim, bool requires_grad)
{
    int size = get_size(shape, ndim);
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->size = size;
    tensor->ndim = ndim;
    tensor->requires_grad = requires_grad;

    tensor->data = NULL;
    tensor->grad = NULL;

    tensor->shape = malloc(ndim * sizeof(int));
    tensor->stride = malloc(ndim * sizeof(int));

    for (int i = ndim; i-- > 0;)
    {
        tensor->shape[i] = shape[i];
        tensor->stride[i] = (i == ndim - 1) ? 1 : tensor->stride[i + 1] * tensor->shape[i + 1];
    }

    tensor->forward_op = NULL;
    tensor->backward_op = NULL;
    return tensor;
}

void tensor_free(tensor_t *tensor)
{
    free(tensor->data);
    free(tensor->grad);

    free(tensor->shape);
    free(tensor->stride);

    if (tensor->forward_op)
    {
        free(tensor->forward_op->sources);
    }
    free(tensor->forward_op);

    if (tensor->backward_op)
    {
        free(tensor->backward_op->targets);
    }
    free(tensor->backward_op);

    free(tensor);
}

tensor_t *tensor(const float data[], int shape[], int ndim, bool requires_grad)
{
    tensor_t *tensor = tensor_create(shape, ndim, requires_grad);
    tensor->forward_op = malloc(sizeof(forward_op_t));
    tensor->forward_op->sources = NULL;
    tensor->forward_op->params = data;
    return tensor;
}

void forward_load(tensor_t *self)
{
    self->data = malloc(self->size * sizeof(float));
    memcpy(self->data, self->forward_op->params, self->size * sizeof(float));
    return;
}

tensor_t *tensor_add(tensor_t *a, tensor_t *b)
{
    tensor_t *c = tensor_create(a->shape, a->ndim, a->requires_grad || b->requires_grad);
    c->forward_op = malloc(sizeof(forward_op_t));
    c->forward_op->sources = malloc(2 * sizeof(tensor_t *));
    c->forward_op->sources[0] = a;
    c->forward_op->sources[1] = b;
    return c;
}

void forward_add(tensor_t *self)
{
    tensor_t *a = self->forward_op->sources[0];
    tensor_t *b = self->forward_op->sources[1];

    self->data = malloc(self->size * sizeof(float));
    for (int i = 0; i < self->size; i++)
    {
        self->data[i] = a->data[i] + b->data[i];
    }
    return;
}

int main()
{
    tensor_t *a = tensor((float[]){1, 2, 3, 4}, (int[]){2, 2}, 2, false);
    tensor_t *b = tensor((float[]){5, 6, 7, 8}, (int[]){2, 2}, 2, false);
    tensor_t *c = tensor_add(a, b);

    forward_load(a);
    forward_load(b);
    forward_add(c);

    for (int i = 0; i < c->size; i++)
    {
        printf("%f ", c->data[i]);
    }
    printf("\n");

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    return 0;
}

// TODO:
// 1. Make forward recursive so that calling on the last tensor propagates the forward call to the first tensor via sources
// 2. Make free recursive so that calling on the last tensor propagates the free call to the first tensor via sources