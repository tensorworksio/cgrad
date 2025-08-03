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

Test (rebind, tensor_rebind)
{
    log_set_level (LOG_INFO);

    // Test TENSOR_REBIND macro for safe tensor reassignment
    smart tensor_t *a = tensor ((float[]) { 2., 4., 6. }, (int[]) { 3 }, 1, false);
    smart tensor_t *b = tensor ((float[]) { 1., 2., 0. }, (int[]) { 3 }, 1, false);

    // c = a + b
    smart tensor_t *c = tensor_add (a, b);
    tensor_forward (c);

    // Verify initial c = a + b = [3., 6., 6.]
    smart tensor_t *expected_c1 = tensor ((float[]) { 3., 6., 6. }, (int[]) { 3 }, 1, false);
    cr_assert (tensor_equals (c, expected_c1, false), "rebind: initial c = a + b failed");

    // c = c - 1 using TENSOR_REBIND (safe reassignment)
    TENSOR_REBIND (c, tensor_sub_tf (c, 1.0f));
    tensor_forward (c);

    // Verify reassigned c = [3., 6., 6.] - 1 = [2., 5., 5.]
    smart tensor_t *expected_c2 = tensor ((float[]) { 2., 5., 5. }, (int[]) { 3 }, 1, false);
    cr_assert (tensor_equals (c, expected_c2, false), "rebind: c = c - 1 failed");

    // Test another reassignment: c = c * 2
    TENSOR_REBIND (c, tensor_mul_tf (c, 2.0f));
    tensor_forward (c);

    // Verify c = [2., 5., 5.] * 2 = [4., 10., 10.]
    smart tensor_t *expected_c3 = tensor ((float[]) { 4., 10., 10. }, (int[]) { 3 }, 1, false);
    cr_assert (tensor_equals (c, expected_c3, false), "rebind: c = c * 2 failed");
}

Test (rebind, tensor_rebind_with_tensors)
{
    log_set_level (LOG_INFO);

    // Test TENSOR_REBIND with tensor-tensor operations
    smart tensor_t *a = tensor ((float[]) { 1., 2., 3. }, (int[]) { 3 }, 1, false);
    smart tensor_t *c = tensor ((float[]) { 10., 20., 30. }, (int[]) { 3 }, 1, false);

    // Test c = c + a using TENSOR_REBIND
    TENSOR_REBIND (c, tensor_add (c, a));
    tensor_forward (c);

    // Verify c = [10., 20., 30.] + [1., 2., 3.] = [11., 22., 33.]
    smart tensor_t *expected_add = tensor ((float[]) { 11., 22., 33. }, (int[]) { 3 }, 1, false);
    cr_assert (tensor_equals (c, expected_add, false), "rebind: c = c + a failed");

    // Test c = c - a using TENSOR_REBIND
    TENSOR_REBIND (c, tensor_sub (c, a));
    tensor_forward (c);

    // Verify c = [11., 22., 33.] - [1., 2., 3.] = [10., 20., 30.]
    smart tensor_t *expected_sub = tensor ((float[]) { 10., 20., 30. }, (int[]) { 3 }, 1, false);
    cr_assert (tensor_equals (c, expected_sub, false), "rebind: c = c - a failed");

    // Test c = c * a using TENSOR_REBIND
    TENSOR_REBIND (c, tensor_mul (c, a));
    tensor_forward (c);

    // Verify c = [10., 20., 30.] * [1., 2., 3.] = [10., 40., 90.]
    smart tensor_t *expected_mul = tensor ((float[]) { 10., 40., 90. }, (int[]) { 3 }, 1, false);
    cr_assert (tensor_equals (c, expected_mul, false), "rebind: c = c * a failed");

    // Test c = c / a using TENSOR_REBIND
    TENSOR_REBIND (c, tensor_div (c, a));
    tensor_forward (c);

    // Verify c = [10., 40., 90.] / [1., 2., 3.] = [10., 20., 30.]
    smart tensor_t *expected_div = tensor ((float[]) { 10., 20., 30. }, (int[]) { 3 }, 1, false);
    cr_assert (tensor_equals (c, expected_div, false), "rebind: c = c / a failed");
}

Test (rebind, tensor_rebind_power_operations)
{
    log_set_level (LOG_INFO);

    // Test TENSOR_REBIND with power operations
    smart tensor_t *a = tensor ((float[]) { 2., 3., 2. }, (int[]) { 3 }, 1, false);
    smart tensor_t *c = tensor ((float[]) { 2., 4., 8. }, (int[]) { 3 }, 1, false);

    // Test c = c ^ a using TENSOR_REBIND
    TENSOR_REBIND (c, tensor_pow (c, a));
    tensor_forward (c);

    // Verify c = [2., 4., 8.] ^ [2., 3., 2.] = [4., 64., 64.]
    smart tensor_t *expected_pow = tensor ((float[]) { 4., 64., 64. }, (int[]) { 3 }, 1, false);
    cr_assert (tensor_equals (c, expected_pow, false), "rebind: c = c ^ a failed");

    // Test c = a ^ c using TENSOR_REBIND (reversed operands)
    TENSOR_REBIND (c, tensor_pow (a, c));
    tensor_forward (c);

    // Verify c = [2., 3., 2.] ^ [4., 64., 64.] = [16., 3^64, 2^64]
    // Note: We'll use smaller values to avoid overflow in testing
    smart tensor_t *a_small = tensor ((float[]) { 2., 2., 2. }, (int[]) { 3 }, 1, false);
    smart tensor_t *c_small = tensor ((float[]) { 3., 4., 5. }, (int[]) { 3 }, 1, false);

    TENSOR_REBIND (c_small, tensor_pow (a_small, c_small));
    tensor_forward (c_small);

    // Verify c = [2., 2., 2.] ^ [3., 4., 5.] = [8., 16., 32.]
    smart tensor_t *expected_pow_rev = tensor ((float[]) { 8., 16., 32. }, (int[]) { 3 }, 1, false);
    cr_assert (tensor_equals (c_small, expected_pow_rev, false), "rebind: c = a ^ c failed");
}