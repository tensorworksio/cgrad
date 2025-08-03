#include "log.h"
#include "tensor.h"
#include <criterion/criterion.h>
#include <math.h>

Test (add, add_tt)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *b   = tensor ((float[]) { 5., 6., 7., 8. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_add (a, b);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 6., 8., 10., 12. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "add_tt failed");
}

Test (add, add_tf)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_add_tf (a, 5.);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 6., 7., 8., 9. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "add_tf failed");
}

Test (add, add_ft)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_add_ft (5., a);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 6., 7., 8., 9. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "add_ft failed");
}

Test (sub, sub_tt)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *b   = tensor ((float[]) { 5., 6., 7., 8. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_sub (a, b);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { -4., -4., -4., -4. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "sub_tt failed");
}

Test (sub, sub_tf)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_sub_tf (a, 5.);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { -4., -3., -2., -1. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "sub_tf failed");
}

Test (sub, sub_ft)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_sub_ft (5., a);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 4., 3., 2., 1. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "sub_ft failed");
}

Test (mul, mul_tt)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *b   = tensor ((float[]) { 5., 6., 7., 8. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_mul (a, b);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 5., 12., 21., 32. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "mul_tt failed");
}

Test (mul, mul_tf)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_mul_tf (a, 5.);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 5., 10., 15., 20. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "mul_tf failed");
}

Test (mul, mul_ft)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_mul_ft (5., a);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 5., 10., 15., 20. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "mul_ft failed");
}

Test (div, div_tt)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 5., 12., 21., 32. }, (int[]) { 4 }, 1, false);
    smart tensor_t *b   = tensor ((float[]) { 5., 6., 7., 8. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_div (a, b);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "div_tt failed");
}

Test (div, div_tf)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 5., 10., 15., 20. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_div_tf (a, 5.);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "div_tf failed");
}

Test (div, div_ft)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 5., 10., 15., 20. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_div_ft (5., a);
    tensor_forward (res);

    smart tensor_t *expected
        = tensor ((float[]) { 1., 0.5, 0.333333, 0.25 }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "div_ft failed");
}

Test (pow, pow_tt)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *b   = tensor ((float[]) { 2., 3., 4., 5. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_pow (a, b);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 1., 8., 81., 1024. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "pow_tt failed");
}

Test (pow, pow_tf)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_pow_tf (a, 2.);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 1., 4., 9., 16. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "pow_tf failed");
}

Test (pow, pow_ft)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_pow_ft (2., a);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 2., 4., 8., 16. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "pow_ft failed");
}

Test (exp, exp_t)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { logf (1.), logf (2.), logf (3.), logf (4.) },
                                  (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_exp (a);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "exp_t failed");
}

Test (neg, neg_t)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., -2., 3., -4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_neg (a);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { -1., 2., -3., 4. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "neg_t failed");
}

Test (relu, relu_t)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., -2., 3., -4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_relu (a);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { 1., 0., 3., 0. }, (int[]) { 4 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "relu_t failed");
}

Test (sum, sum_t)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a   = tensor ((float[]) { 1., -2., 3., -4. }, (int[]) { 4 }, 1, false);
    smart tensor_t *res = tensor_sum (a);
    tensor_forward (res);

    smart tensor_t *expected = tensor ((float[]) { -2. }, (int[]) { 1 }, 1, false);
    cr_assert (tensor_equals (res, expected, true), "sum_t failed");
}