#include <criterion/criterion.h>
#include <math.h>
#include "tensor.h"
#include "log.h"

// BINARY OPS
Test(add, tensor_add)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t *b = tensor((float[]){5., 6., 7., 8.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_add(a, b);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){6., 8., 10., 12.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_add failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(add, tensor_add_tf)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_add_tf(a, 5.);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){6., 7., 8., 9.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_add_tf failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(add, tensor_add_ft)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_add_ft(5., a);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){6., 7., 8., 9.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_add_ft failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(sub, tensor_sub)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t *b = tensor((float[]){5., 6., 7., 8.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_sub(a, b);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){-4., -4., -4., -4.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_sub failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(sub, tensor_sub_tf)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_sub_tf(a, 5.);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){-4., -3., -2., -1.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_sub_tf failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(sub, tensor_sub_ft)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_sub_ft(5., a);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){4., 3., 2., 1.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_sub_ft failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(mul, tensor_mul)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t *b = tensor((float[]){5., 6., 7., 8.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_mul(a, b);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){5., 12., 21., 32.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_mul failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(mul, tensor_mul_tf)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_mul_tf(a, 5.);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){5., 10., 15., 20.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_mul_tf failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(mul, tensor_mul_ft)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_mul_ft(5., a);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){5., 10., 15., 20.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_mul_ft failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(div, tensor_div)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){5., 12., 21., 32.}, (int[]){4}, 1, false);
    tensor_t *b = tensor((float[]){5., 6., 7., 8.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_div(a, b);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_div failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(div, tensor_div_tf)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){5., 10., 15., 20.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_div_tf(a, 5.);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_div_tf failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(div, tensor_div_ft)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){5., 10., 15., 20.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_div_ft(5., a);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){1., 0.5, 0.333333, 0.25}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_div_ft failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(pow, tensor_pow)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t *b = tensor((float[]){2., 3., 4., 5.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_pow(a, b);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){1., 8., 81., 1024.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_pow failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(pow, tensor_pow_tf)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_pow_tf(a, 2.);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){1., 4., 9., 16.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_pow_tf failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(pow, tensor_pow_ft)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_pow_ft(2., a);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){2., 4., 8., 16.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_pow_ft failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

// UNARY OPS
Test(exp, tensor_exp)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){logf(1.), logf(2.), logf(3.), logf(4.)}, (int[]){4}, 1, false);
    tensor_t *res = tensor_exp(a);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_exp failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(neg, tensor_neg)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., -2., 3., -4.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_neg(a);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){-1., 2., -3., 4.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_neg failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(relu, tensor_relu)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., -2., 3., -4.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_relu(a);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){1., 0., 3., 0.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_relu failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

// REDUCE OPS
Test(sum, tensor_sum)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., -2., 3., -4.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_sum(a);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){-2.}, (int[]){1}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_sum failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

// MOVEMENT OPS
Test(reshape, tensor_reshape)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t *res = tensor_reshape(a, (int[]){2, 2}, 2);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){1., 2., 3., 4.}, (int[]){2, 2}, 2, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_reshape failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(transpose, tensor_transpose)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4., 5., 6.}, (int[]){3, 2}, 2, false);
    tensor_t *res = tensor_transpose(a, 0, 1);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){1., 3., 5., 2., 4., 6.}, (int[]){2, 3}, 2, false);
    cr_assert(tensor_equals(res, expected, false), "tensor_transpose failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(slice, tensor_slice)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4., 5., 6., 7., 8.}, (int[]){4, 2}, 2, false);
    tensor_t *res = tensor_slice(a, (slice_t[]){(slice_t){1, 3, 1}, (slice_t){0, 2, 1}});
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){3., 4., 5., 6.}, (int[]){2, 2}, 2, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_slice failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(cat, tensor_cat_0)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){2, 2}, 2, false);
    tensor_t *b = tensor((float[]){5., 6., 7., 8.}, (int[]){2, 2}, 2, false);
    tensor_t *res = tensor_cat((tensor_t *[]){a, b}, 2, 0);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){1., 2., 3., 4., 5., 6., 7., 8.}, (int[]){4, 2}, 2, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_cat_0 failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(cat, tensor_cat_1)
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){1., 2., 3., 4.}, (int[]){2, 2}, 2, false);
    tensor_t *b = tensor((float[]){5., 6., 7., 8.}, (int[]){2, 2}, 2, false);
    tensor_t *res = tensor_cat((tensor_t *[]){a, b}, 2, 1);
    tensor_forward(res);

    tensor_t *expected = tensor((float[]){1., 2., 5., 6., 3., 4., 7., 8.}, (int[]){2, 4}, 2, false);
    cr_assert(tensor_equals(res, expected, true), "tensor_cat_1 failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}