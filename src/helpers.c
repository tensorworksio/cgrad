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

bool is_same_shape(int shape_a[], int shape_b[], int ndim_a, int ndim_b) {
    if (ndim_a != ndim_b) return false;

    for (int i = 0; i < ndim_a; i++)
    {
        if (shape_a[i] != shape_b[i]) return false;
    }
    return true;
}

bool is_equal_data(float* data_a, float* data_b, int size) {
    if(data_a == NULL || data_b == NULL) return false;
    for (int i = 0; i < size; i++)
    {
        if (fabs(data_a[i] - data_b[i]) > EPSILON) return false;
    }
    return true;
}

void set_data(float* data, float value, int size) {
    for (int i = 0; i < size; i++)
    {
        data[i] = value;
    }
}

void print_data(float* data, int shape[], int ndim)
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