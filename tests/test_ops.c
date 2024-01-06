#include <criterion/criterion.h>
#include "tensor.h"
#include "helpers.h"
#include "ops.h"

Test(add, add_tt)
{
    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t* b = tensor((float[]){5., 6., 7., 8.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_add(a, b);
    tensor_t* expected = tensor((float[]){6., 8., 10., 12.}, (int[]){4}, 1, false);
    
    cr_assert(tensor_equals(res, expected, true), "add_tt failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(add, add_tf)
{
    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_add_tf(a, 5.);
    tensor_t* expected = tensor((float[]){6., 7., 8., 9.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "add_tf failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(add, add_ft)
{
    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_add_ft(5., a);
    tensor_t* expected = tensor((float[]){6., 7., 8., 9.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "add_ft failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(sub, sub_tt)
{
    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t* b = tensor((float[]){5., 6., 7., 8.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_sub(a, b);
    tensor_t* expected = tensor((float[]){-4., -4., -4., -4.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "sub_tt failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(sub, sub_tf)
{
    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_sub_tf(a, 5.);
    tensor_t* expected = tensor((float[]){-4., -3., -2., -1.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "sub_tf failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(sub, sub_ft)
{
    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_sub_ft(5., a);
    tensor_t* expected = tensor((float[]){4., 3., 2., 1.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "sub_ft failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(mul, mul_tt)
{
    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t* b = tensor((float[]){5., 6., 7., 8.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_mul(a, b);
    tensor_t* expected = tensor((float[]){5., 12., 21., 32.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "mul_tt failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(mul, mul_tf)
{
    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_mul_tf(a, 5.);
    tensor_t* expected = tensor((float[]){5., 10., 15., 20.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "mul_tf failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(mul, mul_ft)
{
    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_mul_ft(5., a);
    tensor_t* expected = tensor((float[]){5., 10., 15., 20.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "mul_ft failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(div, div_tt)
{
    tensor_t* a = tensor((float[]){5., 12., 21., 32.}, (int[]){4}, 1, false);
    tensor_t* b = tensor((float[]){5., 6., 7., 8.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_div(a, b);
    tensor_t* expected = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "div_tt failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(div, div_tf)
{
    tensor_t* a = tensor((float[]){5., 10., 15., 20.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_div_tf(a, 5.);
    tensor_t* expected = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "div_tf failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(div, div_ft)
{
    tensor_t* a = tensor((float[]){5., 10., 15., 20.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_div_ft(5., a);
    tensor_t* expected = tensor((float[]){1., 0.5, 0.333333, 0.25}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "div_ft failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(pow, pow_tt)
{
    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t* b = tensor((float[]){2., 3., 4., 5.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_pow(a, b);
    tensor_t* expected = tensor((float[]){1., 8., 81., 1024.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "pow_tt failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(pow, pow_tf)
{
    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_pow_tf(a, 2.);
    tensor_t* expected = tensor((float[]){1., 4., 9., 16.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "pow_tf failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(pow, pow_ft)
{
    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_pow_ft(2., a);
    tensor_t* expected = tensor((float[]){2., 4., 8., 16.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "pow_ft failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(exp, exp_t)
{
    tensor_t* a = tensor((float[]){1., 2., 3., 4.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_exp(a);
    tensor_t* expected = tensor((float[]){expf(1.), expf(2.), expf(3.), expf(4.)}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "exp_t failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(neg, neg_t)
{
    tensor_t* a = tensor((float[]){1., -2., 3., -4.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_neg(a);
    tensor_t* expected = tensor((float[]){-1., 2., -3., 4.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "neg_t failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(relu, relu_t)
{
    tensor_t* a = tensor((float[]){1., -2., 3., -4.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_relu(a);
    tensor_t* expected = tensor((float[]){1., 0., 3., 0.}, (int[]){4}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "relu_t failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}

Test(sum, sum_t)
{
    tensor_t* a = tensor((float[]){1., -2., 3., -4.}, (int[]){4}, 1, false);
    tensor_t* res = tensor_sum(a);
    tensor_t* expected = tensor((float[]){-2.}, (int[]){1}, 1, false);
    cr_assert(tensor_equals(res, expected, true), "sum_t failed");
    tensor_free(res, true);
    tensor_free(expected, true);
}