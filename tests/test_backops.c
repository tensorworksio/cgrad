#include <criterion/criterion.h>
#include "tensor.h"
#include "helpers.h"
#include "ops.h"
#include "log.h"

Test(add, backward_add)
{   
    log_set_level(LOG_INFO);

    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, true);
    tensor_t* b = tensor((float[]){5., 6., 7., 8.}, (int[]){4}, 1, true);
    tensor_t* c = tensor_add(a, b);
    tensor_t* y = tensor_sum(c);

    tensor_backward(y);

    tensor_t* a_grad = tensor_create(a->shape, a->ndim, true);
    tensor_set_data(a_grad, a->data, a->size);
    tensor_set_grad(a_grad, (float[]){1., 1., 1., 1.}, a->size);

    tensor_t* b_grad = tensor_create(b->shape, b->ndim, true);
    tensor_set_data(b_grad, b->data, b->size);
    tensor_set_grad(b_grad, (float[]){1., 1., 1., 1.}, b->size);

    cr_assert(tensor_equals(a, a_grad, true), "backward_add failed");
    cr_assert(tensor_equals(b, b_grad, true), "backward_add failed");

    tensor_free(y, true);
    tensor_free(a_grad, true);
    tensor_free(b_grad, true);
}

Test(add, backward_add_tf)
{   
    log_set_level(LOG_INFO);

    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, true);
    tensor_t* c = tensor_add_tf(a, 5.);
    tensor_t* y = tensor_sum(c);

    tensor_backward(y);
    
    tensor_t* a_grad = tensor_create(a->shape, a->ndim, true);
    tensor_set_data(a_grad, a->data, a->size);
    tensor_set_grad(a_grad, (float[]){1., 1., 1., 1.}, a->size);

    cr_assert(tensor_equals(a, a_grad, true), "backward_add_tf failed");

    tensor_free(y, true);
    tensor_free(a_grad, true);
}

Test(add, backward_add_ft)
{   
    log_set_level(LOG_INFO);

    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, true);
    tensor_t* c = tensor_add_ft(5., a);
    tensor_t* y = tensor_sum(c);

    tensor_backward(y);
    
    tensor_t* a_grad = tensor_create(a->shape, a->ndim, true);
    tensor_set_data(a_grad, a->data, a->size);
    tensor_set_grad(a_grad, (float[]){1., 1., 1., 1.}, a->size);

    cr_assert(tensor_equals(a, a_grad, true), "backward_add_ft failed");

    tensor_free(y, true);
    tensor_free(a_grad, true);
}

Test(sub, backward_sub)
{   
    log_set_level(LOG_INFO);

    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, true);
    tensor_t* b = tensor((float[]){5., 6., 7., 8.}, (int[]){4}, 1, true);
    tensor_t* c = tensor_sub(a, b);
    tensor_t* y = tensor_sum(c);

    tensor_backward(y);
    
    tensor_t* a_grad = tensor_create(a->shape, a->ndim, true);
    tensor_set_data(a_grad, a->data, a->size);
    tensor_set_grad(a_grad, (float[]){1., 1., 1., 1.}, a->size);

    tensor_t* b_grad = tensor_create(b->shape, b->ndim, true);
    tensor_set_data(b_grad, b->data, b->size);
    tensor_set_grad(b_grad, (float[]){-1., -1., -1., -1.}, b->size);

    cr_assert(tensor_equals(a, a_grad, true), "backward_sub failed");
    cr_assert(tensor_equals(b, b_grad, true), "backward_sub failed");

    tensor_free(y, true);
    tensor_free(a_grad, true);
    tensor_free(b_grad, true);
}

Test(sub, backward_sub_tf)
{   
    log_set_level(LOG_INFO);

    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, true);
    tensor_t* c = tensor_sub_tf(a, 5.);
    tensor_t* y = tensor_sum(c);

    tensor_backward(y);
    
    tensor_t* a_grad = tensor_create(a->shape, a->ndim, true);
    tensor_set_data(a_grad, a->data, a->size);
    tensor_set_grad(a_grad, (float[]){1., 1., 1., 1.}, a->size);

    cr_assert(tensor_equals(a, a_grad, true), "backward_sub_tf failed");

    tensor_free(y, true);
    tensor_free(a_grad, true);
}

Test(sub, backward_sub_ft)
{   
    log_set_level(LOG_INFO);

    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, true);
    tensor_t* c = tensor_sub_ft(5., a);
    tensor_t* y = tensor_sum(c);

    tensor_backward(y);
    
    tensor_t* a_grad = tensor_create(a->shape, a->ndim, true);
    tensor_set_data(a_grad, a->data, a->size);
    tensor_set_grad(a_grad, (float[]){-1., -1., -1., -1.}, a->size);

    cr_assert(tensor_equals(a, a_grad, true), "backward_sub_ft failed");

    tensor_free(y, true);
    tensor_free(a_grad, true);
}

Test(mul, backward_mul)
{   
    log_set_level(LOG_INFO);

    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, true);
    tensor_t* b = tensor((float[]){5., 6., 7., 8.}, (int[]){4}, 1, true);
    tensor_t* c = tensor_mul(a, b);
    tensor_t* y = tensor_sum(c);

    tensor_backward(y);
    
    tensor_t* a_grad = tensor_create(a->shape, a->ndim, true);
    tensor_set_data(a_grad, a->data, a->size);
    tensor_set_grad(a_grad, b->data, b->size);

    tensor_t* b_grad = tensor_create(b->shape, b->ndim, true);
    tensor_set_data(b_grad, b->data, b->size);
    tensor_set_grad(b_grad, a->data, a->size);

    cr_assert(tensor_equals(a, a_grad, true), "backward_mul failed");
    cr_assert(tensor_equals(b, b_grad, true), "backward_mul failed");

    tensor_free(y, true);
    tensor_free(a_grad, true);
    tensor_free(b_grad, true);
}

Test(mul, backward_mul_tf)
{   
    log_set_level(LOG_INFO);

    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, true);

    tensor_t* c = tensor_mul_tf(a, 5.);
    tensor_t* y = tensor_sum(c);

    tensor_backward(y);
    
    tensor_t* a_grad = tensor_create(a->shape, a->ndim, true);
    tensor_set_data(a_grad, a->data, a->size);
    tensor_set_grad(a_grad, (float[]){5., 5., 5., 5.}, a->size);

    cr_assert(tensor_equals(a, a_grad, true), "backward_mul_tf failed");

    tensor_free(y, true);
    tensor_free(a_grad, true);
}

Test(mul, backward_mul_ft)
{   
    log_set_level(LOG_INFO);

    tensor_t* a = tensor((float[]){5., 6., 7., 8.}, (int[]){4}, 1, true);

    tensor_t* c = tensor_mul_ft(5., a);
    tensor_t* y = tensor_sum(c);

    tensor_backward(y);
    
    tensor_t* a_grad = tensor_create(a->shape, a->ndim, true);
    tensor_set_data(a_grad, a->data, a->size);
    tensor_set_grad(a_grad, (float[]){5., 5., 5., 5.}, a->size);

    cr_assert(tensor_equals(a, a_grad, true), "backward_mul_ft failed");

    tensor_free(y, true);
    tensor_free(a_grad, true);
}