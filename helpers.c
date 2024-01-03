#include "helpers.h"

int get_size(int shape[], int ndim)
{
    int size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size *= shape[i];
    }
    return size;
}

bool same_shape(tensor_t* a, tensor_t* b)
{
    if (a->ndim != b->ndim)
    {
        return false;
    }
    for (int i = 0; i < a->ndim; i++)
    {
        if (a->shape[i] != b->shape[i])
        {
            return false;
        }
    }
    return true;
}

void set_tensor_data(float* data, int size, float value)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = value;
    }
}

void print_tensor_data(float* data, int shape[], int ndim)
{
    int size = get_size(shape, ndim);
    int EOD[ndim];
    for (int d = ndim-1; d >= 0; --d) {
        if (d == ndim-1) {
            EOD[d] = shape[d];
        } else {
            EOD[d] = EOD[d+1] * shape[d];
        }
    }
    int idx = 0;
    while (idx < size) {
        for (int d = 0; d < ndim; ++d) {
            if (idx % EOD[d] == 0) printf("[");
        }
        printf("%f ", data[idx]);
        for (int d = 0; d < ndim; ++d) {
            if ((idx + 1) % EOD[d] == 0) printf("]");
        }
        for (int d = 0; d < ndim; ++d) {
            if ((idx + 1) % EOD[d] == 0) {
                printf("\n");
                break;
            }
        }
        idx++;
    }
}