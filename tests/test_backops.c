#include "log.h"
#include "tensor.h"
#include <criterion/criterion.h>
#include <math.h>

Test (add, backward_add)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *b = tensor ((float[]) { 5., 6., 7., 8. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_add (a, b);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 1., 1., 1., 1. }, a->size);

    smart tensor_t *b_grad = tensor_create (b->shape, b->ndim, true);
    tensor_set_data (b_grad, b->data, b->size);
    tensor_set_grad (b_grad, (float[]) { 1., 1., 1., 1. }, b->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_add failed");
    cr_assert (tensor_equals (b, b_grad, true), "backward_add failed");
}

Test (add, backward_add_tf)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_add_tf (a, 5.);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 1., 1., 1., 1. }, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_add_tf failed");
}

Test (add, backward_add_ft)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_add_ft (5., a);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 1., 1., 1., 1. }, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_add_ft failed");
}

Test (sub, backward_sub)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *b = tensor ((float[]) { 5., 6., 7., 8. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_sub (a, b);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 1., 1., 1., 1. }, a->size);

    smart tensor_t *b_grad = tensor_create (b->shape, b->ndim, true);
    tensor_set_data (b_grad, b->data, b->size);
    tensor_set_grad (b_grad, (float[]) { -1., -1., -1., -1. }, b->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_sub failed");
    cr_assert (tensor_equals (b, b_grad, true), "backward_sub failed");
}

Test (sub, backward_sub_tf)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_sub_tf (a, 5.);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 1., 1., 1., 1. }, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_sub_tf failed");
}

Test (sub, backward_sub_ft)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_sub_ft (5., a);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { -1., -1., -1., -1. }, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_sub_ft failed");
}

Test (mul, backward_mul)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *b = tensor ((float[]) { 5., 6., 7., 8. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_mul (a, b);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, b->data, b->size);

    smart tensor_t *b_grad = tensor_create (b->shape, b->ndim, true);
    tensor_set_data (b_grad, b->data, b->size);
    tensor_set_grad (b_grad, a->data, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_mul failed");
    cr_assert (tensor_equals (b, b_grad, true), "backward_mul failed");
}

Test (mul, backward_mul_tf)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_mul_tf (a, 5.);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 5., 5., 5., 5. }, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_mul_tf failed");
}

Test (mul, backward_mul_ft)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 5., 6., 7., 8. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_mul_ft (5., a);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 5., 5., 5., 5. }, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_mul_ft failed");
}

Test (div, backward_div_tt)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 5., 6., 9., 8. }, (int[]) { 4 }, 1, true);
    smart tensor_t *b = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_div (a, b);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 1., 0.5, 1. / 3, 0.25 }, a->size);

    smart tensor_t *b_grad = tensor_create (b->shape, b->ndim, true);
    tensor_set_data (b_grad, b->data, b->size);
    tensor_set_grad (b_grad, (float[]) { -5., -1.5, -1., -0.5 }, b->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_div failed");
    cr_assert (tensor_equals (b, b_grad, true), "backward_div failed");
}

Test (div, backward_div_tf)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 5., 6., 9., 8. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_div_tf (a, 5.);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 0.2, 0.2, 0.2, 0.2 }, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_div_tf failed");
}

Test (div, backward_div_ft)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 5., 6., 9., 8. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_div_ft (5., a);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { -0.2, -0.1388889, -0.0617284, -0.078125 }, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_div_ft failed");
}

Test (pow, backward_pow_tt)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 4., 3., 2., 1. }, (int[]) { 4 }, 1, true);
    smart tensor_t *b = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_pow (a, b);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, b->size);
    tensor_set_grad (a_grad, (float[]) { 1., 6., 12., 4. }, a->size);

    smart tensor_t *b_grad = tensor_create (b->shape, b->ndim, true);
    tensor_set_data (b_grad, b->data, a->size);
    tensor_set_grad (b_grad, (float[]) { 5.545177, 9.88751, 5.545177, 0. }, b->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_pow failed");
    cr_assert (tensor_equals (b, b_grad, true), "backward_pow failed");
}

Test (pow, backward_pow_tf)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 4., 3., 2., 1. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_pow_tf (a, 5.);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 1280., 405., 80., 5. }, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_pow_tf failed");
}

Test (pow, backward_pow_ft)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 4., 3., 2., 1. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_pow_ft (5., a);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 1005.898743, 201.179749, 40.235947, 8.047190 }, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_pow_ft failed");
}

Test (exp, backward_exp)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a
        = tensor ((float[]) { logf (1.), logf (2.), logf (3.), logf (4.) }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_exp (a);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 1., 2., 3., 4. }, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_exp failed");
}

Test (relu, backward_relu)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { -1., 2., -3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *c = tensor_relu (a);
    smart tensor_t *y = tensor_sum (c);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 0., 1., 0., 1. }, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_relu failed");
}

Test (sum, backward_sum)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { -1., 2., -3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *y = tensor_sum (a);

    tensor_backward (y);

    smart tensor_t *a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (a_grad, a->data, a->size);
    tensor_set_grad (a_grad, (float[]) { 1., 1., 1., 1. }, a->size);

    cr_assert (tensor_equals (a, a_grad, true), "backward_sum failed");
}

Test (complex, diamond_shape_graph)
{
    log_set_level (LOG_INFO);

    // Create a diamond-shape computational graph similar to main.c
    // This tests that smart pointers properly handle shared references
    // without double-free errors
    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4., 5., 6. }, (int[]) { 3, 2 }, 2, true);
    smart tensor_t *b = tensor_mul_ft (2.f, a); // b = 2 * a
    smart tensor_t *c = tensor_mul_ft (3.f, a); // c = 3 * a (a is shared between b and c)
    smart tensor_t *d = tensor_add (b, c);      // d = b + c
    smart tensor_t *e = tensor_sum (d);         // e = sum(d)

    tensor_backward (e); // compute gradients

    // Verify the computation results
    // a = [1,2,3,4,5,6] reshaped to (3,2)
    // b = 2*a = [2,4,6,8,10,12]
    // c = 3*a = [3,6,9,12,15,18]
    // d = b+c = [5,10,15,20,25,30]
    // e = sum(d) = 105

    smart tensor_t *expected_e = tensor ((float[]) { 105. }, (int[]) { 1 }, 1, false);
    cr_assert (tensor_equals (e, expected_e, false), "diamond_shape_graph: final result incorrect");

    // Verify gradients
    // de/da = de/dd * dd/da = 1 * (db/da + dc/da) = 1 * (2 + 3) = 5 for each element
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 5., 5., 5., 5., 5., 5. }, a->size);

    cr_assert (tensor_equals (a, expected_a_grad, true),
               "diamond_shape_graph: gradient computation failed");
}