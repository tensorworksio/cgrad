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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (c, NULL, 0);

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
    smart tensor_t *y = tensor_sum (a, NULL, 0);

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
    smart tensor_t *b = tensor_mul_ft (2.f, a);  // b = 2 * a
    smart tensor_t *c = tensor_mul_ft (3.f, a);  // c = 3 * a (a is shared between b and c)
    smart tensor_t *d = tensor_add (b, c);       // d = b + c
    smart tensor_t *e = tensor_sum (d, NULL, 0); // e = sum(d)

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

Test (complex, diamond_slice_reshape)
{
    log_set_level (LOG_INFO);

    // 2d example from main.c
    // sum axis 0
    smart tensor_t *trow = tensor_zeros ((int[]) { 3 }, 1, false);
    smart tensor_t *tin  = tensor ((float[]) { 1., 2., 3., 4., 5., 6. }, (int[]) { 2, 3 }, 2, true);

    smart tensor_t *row0 = tensor_slice (tin, (slice_t[]) { SLICE_ONE (0), SLICE_ALL });
    TENSOR_REBIND (row0, tensor_reshape (row0, (int[]) { 3 }, 1));

    smart tensor_t *row1 = tensor_slice (tin, (slice_t[]) { SLICE_ONE (1), SLICE_ALL });
    TENSOR_REBIND (row1, tensor_reshape (row1, (int[]) { 3 }, 1));

    TENSOR_REBIND (trow, tensor_add (trow, row0));
    TENSOR_REBIND (trow, tensor_add (trow, row1));

    smart tensor_t *out = tensor_sum (trow, NULL, 0);

    tensor_backward (out);

    smart tensor_t *expected_grad = tensor_create (tin->shape, tin->ndim, true);
    tensor_set_data (expected_grad, tin->data, tin->size);
    tensor_set_grad (expected_grad, (float[]) { 1., 1., 1., 1., 1., 1. }, tin->size);

    cr_assert (tensor_equals (tin, expected_grad, true),
               "diamond_slice_reshape: tin gradients incorrect");
}

Test (rebind, backward_rebind_basic)
{
    log_set_level (LOG_INFO);

    // Test backward propagation with TENSOR_REBIND for scalar operations
    smart tensor_t *a = tensor ((float[]) { 2., 3., 4. }, (int[]) { 3 }, 1, true);
    smart tensor_t *c = tensor_mul_tf (a, 2.0f); // c = 2 * a = [4., 6., 8.]

    // c = c - 1 using TENSOR_REBIND
    TENSOR_REBIND (c, tensor_sub_tf (c, 1.0f)); // c = c - 1 = [3., 5., 7.]

    // Final operation: sum for scalar output
    smart tensor_t *y = tensor_sum (c, NULL, 0); // y = sum(c) = 15

    tensor_backward (y);

    // Expected gradients: dy/da = dy/dc * dc/da = 1 * 2 = 2 for each element
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 2., 2., 2. }, a->size);

    cr_assert (tensor_equals (a, expected_a_grad, true), "backward_rebind_basic failed");
}

Test (rebind, backward_rebind_tensor_ops)
{
    log_set_level (LOG_INFO);

    // Test backward propagation with TENSOR_REBIND for tensor-tensor operations
    smart tensor_t *a = tensor ((float[]) { 1., 2., 3. }, (int[]) { 3 }, 1, true);
    smart tensor_t *b = tensor ((float[]) { 2., 3., 4. }, (int[]) { 3 }, 1, true);
    smart tensor_t *c = tensor_mul (a, b); // c = a * b = [2., 6., 12.]

    // c = c + a using TENSOR_REBIND
    TENSOR_REBIND (c, tensor_add (c, a)); // c = c + a = [3., 8., 15.]

    // Final operation: sum for scalar output
    smart tensor_t *y = tensor_sum (c, NULL, 0); // y = sum(c) = 26

    tensor_backward (y);

    // Expected gradients for a:
    // dy/da = dy/dc * (dc/da_mul + dc/da_add) = 1 * (b + 1) = [3., 4., 5.]
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 3., 4., 5. }, a->size);

    // Expected gradients for b:
    // dy/db = dy/dc * dc/db = 1 * a = [1., 2., 3.]
    smart tensor_t *expected_b_grad = tensor_create (b->shape, b->ndim, true);
    tensor_set_data (expected_b_grad, b->data, b->size);
    tensor_set_grad (expected_b_grad, (float[]) { 1., 2., 3. }, b->size);

    cr_assert (tensor_equals (a, expected_a_grad, true),
               "backward_rebind_tensor_ops: a gradient failed");
    cr_assert (tensor_equals (b, expected_b_grad, true),
               "backward_rebind_tensor_ops: b gradient failed");
}

Test (rebind, backward_rebind_complex_chain)
{
    log_set_level (LOG_INFO);

    // Test backward propagation with multiple TENSOR_REBIND operations
    // This reproduces the pattern from main.c
    smart tensor_t *a = tensor ((float[]) { 2., 4., 6. }, (int[]) { 3 }, 1, true);
    smart tensor_t *b = tensor ((float[]) { 1., 2., 0. }, (int[]) { 3 }, 1, true);

    // c = a + b
    smart tensor_t *c = tensor_add (a, b); // c = [3., 6., 6.]

    // c = c - 1 using TENSOR_REBIND
    TENSOR_REBIND (c, tensor_sub_tf (c, 1.0f)); // c = [2., 5., 5.]

    // d = c ^ 3
    smart tensor_t *d = tensor_pow_tf (c, 3.); // d = [8., 125., 125.]

    // e = relu(d) (all positive, so no change)
    smart tensor_t *e = tensor_relu (d); // e = [8., 125., 125.]

    // f = sum(e)
    smart tensor_t *f = tensor_sum (e, NULL, 0); // f = 258

    tensor_backward (f);

    // Expected gradients for a:
    // df/da = df/de * de/dd * dd/dc * dc/da = 1 * 1 * 3*c^2 * 1 = 3 * [4., 25., 25.] =
    // [12., 75., 75.]
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 12., 75., 75. }, a->size);

    // Expected gradients for b: same as a since dc/db = 1
    smart tensor_t *expected_b_grad = tensor_create (b->shape, b->ndim, true);
    tensor_set_data (expected_b_grad, b->data, b->size);
    tensor_set_grad (expected_b_grad, (float[]) { 12., 75., 75. }, b->size);

    cr_assert (tensor_equals (a, expected_a_grad, true),
               "backward_rebind_complex_chain: a gradient failed");
    cr_assert (tensor_equals (b, expected_b_grad, true),
               "backward_rebind_complex_chain: b gradient failed");
}

Test (rebind, backward_rebind_power_ops)
{
    log_set_level (LOG_INFO);

    // Test backward propagation with TENSOR_REBIND for power operations
    smart tensor_t *a = tensor ((float[]) { 2., 3. }, (int[]) { 2 }, 1, true);
    smart tensor_t *c = tensor_pow_tf (a, 2.); // c = a^2 = [4., 9.]

    // c = c ^ a using TENSOR_REBIND (where a = [2., 3.])
    TENSOR_REBIND (c, tensor_pow (c, a)); // c = [4., 9.] ^ [2., 3.] = [16., 729.]

    // Final operation: sum for scalar output
    smart tensor_t *y = tensor_sum (c, NULL, 0); // y = sum(c) = 745

    tensor_backward (y);

    // Expected gradients for a:
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 54.18071, 3059.7769 }, a->size);

    cr_assert (tensor_equals (a, expected_a_grad, true),
               "backward_rebind_power_ops: a gradient failed");
}

Test (clone, backward_tensor_clone_basic)
{
    log_set_level (LOG_INFO);

    // Test backward propagation through tensor_clone
    smart tensor_t *a      = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *cloned = tensor_clone (a);
    smart tensor_t *y      = tensor_sum (cloned, NULL, 0);

    tensor_backward (y);

    // Expected gradients: dy/da = dy/dcloned * dcloned/da = 1 * 1 = 1 for each element
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 1., 1., 1., 1. }, a->size);

    cr_assert (tensor_equals (a, expected_a_grad, true), "backward_tensor_clone_basic failed");
}

Test (clone, backward_tensor_clone_chain)
{
    log_set_level (LOG_INFO);

    // Test backward propagation through tensor_clone in a computation chain
    smart tensor_t *a      = tensor ((float[]) { 2., 3., 4. }, (int[]) { 3 }, 1, true);
    smart tensor_t *cloned = tensor_clone (a);
    smart tensor_t *scaled = tensor_mul_tf (cloned, 2.0f); // scaled = 2 * cloned = 2 * a
    smart tensor_t *y      = tensor_sum (scaled, NULL, 0); // y = sum(2 * a) = 2 * sum(a)

    tensor_backward (y);

    // Expected gradients: dy/da = dy/dscaled * dscaled/dcloned * dcloned/da = 1 * 2 * 1 = 2
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 2., 2., 2. }, a->size);

    cr_assert (tensor_equals (a, expected_a_grad, true), "backward_tensor_clone_chain failed");
}

Test (clone, backward_tensor_clone_multiple)
{
    log_set_level (LOG_INFO);

    // Test backward propagation with multiple clones of the same tensor
    smart tensor_t *a      = tensor ((float[]) { 1., 2., 3. }, (int[]) { 3 }, 1, true);
    smart tensor_t *clone1 = tensor_clone (a);
    smart tensor_t *clone2 = tensor_clone (a);

    // Use both clones in computation: y = sum(clone1 + clone2) = sum(2*a)
    smart tensor_t *added = tensor_add (clone1, clone2);
    smart tensor_t *y     = tensor_sum (added, NULL, 0);

    tensor_backward (y);

    // Expected gradients: dy/da = dy/dadded * (dadded/dclone1 * dclone1/da + dadded/dclone2 *
    // dclone2/da)
    //                    = 1 * (1 * 1 + 1 * 1) = 2 for each element
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 2., 2., 2. }, a->size);

    cr_assert (tensor_equals (a, expected_a_grad, true), "backward_tensor_clone_multiple failed");
}

Test (clone, backward_tensor_clone_main_example)
{
    log_set_level (LOG_INFO);

    // Test the exact pattern from main.c
    smart tensor_t *a
        = tensor ((float[]) { 1., 2., 3., 4., 5., 6., 7., 8. }, (int[]) { 2, 2, 2 }, 3, true);
    smart tensor_t *b = tensor_clone (a);
    smart tensor_t *c = tensor_sum (b, NULL, 0);

    tensor_backward (c);

    // Expected gradients: dc/da = dc/db * db/da = 1 * 1 = 1 for each element
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 1., 1., 1., 1., 1., 1., 1., 1. }, a->size);

    cr_assert (tensor_equals (a, expected_a_grad, true),
               "backward_tensor_clone_main_example failed");
}

Test (reshape, backward_reshape)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 4 }, 1, true);
    smart tensor_t *b = tensor_reshape (a, (int[]) { 2, 2 }, 2);
    smart tensor_t *y = tensor_sum (b, NULL, 0);

    tensor_backward (y);

    // Since reshape shares grad, a's grad should be set
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 1., 1., 1., 1. }, a->size);

    cr_assert (tensor_equals (a, expected_a_grad, true), "backward_reshape failed");
}

Test (transpose, backward_transpose)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 2, 2 }, 2, true);
    smart tensor_t *b = tensor_transpose (a, 0, 1);
    smart tensor_t *y = tensor_sum (b, NULL, 0);

    tensor_backward (y);

    // Since transpose shares grad, a's grad should be set
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 1., 1., 1., 1. }, a->size);

    cr_assert (tensor_equals (a, expected_a_grad, true), "backward_transpose failed");
}

Test (slice, backward_slice)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a
        = tensor ((float[]) { 1., 2., 3., 4., 5., 6., 7., 8. }, (int[]) { 4, 2 }, 2, true);
    smart tensor_t *b
        = tensor_slice (a, (slice_t[]) { (slice_t) { 1, 3, 1 }, (slice_t) { 0, 2, 1 } });
    smart tensor_t *y = tensor_sum (b, NULL, 0);

    tensor_backward (y);

    // Expected grad for a: zeros except positions 2,3,4,5 (0-indexed) which are 1
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 0., 0., 1., 1., 1., 1., 0., 0. }, a->size);

    cr_assert (tensor_equals (a, expected_a_grad, true), "backward_slice failed");
}

Test (cat, cat_axis0_backward)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 2, 2 }, 2, true);
    smart tensor_t *b = tensor ((float[]) { 5., 6., 7., 8. }, (int[]) { 2, 2 }, 2, true);
    smart tensor_t *c = tensor_cat ((tensor_t *[]) { a, b }, 2, 0);
    smart tensor_t *weights
        = tensor ((float[]) { 1., 2., 3., 4., 5., 6., 7., 8. }, (int[]) { 4, 2 }, 2, false);
    smart tensor_t *loss = tensor_mul (c, weights);
    smart tensor_t *y    = tensor_sum (loss, NULL, 0);

    tensor_backward (y);

    // Expected grads: for a (rows 0-1), grad_a = weights[0:4] = [1,2,3,4]
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 1., 2., 3., 4. }, a->size);

    // For b (rows 2-3), grad_b = weights[4:8] = [5,6,7,8]
    smart tensor_t *expected_b_grad = tensor_create (b->shape, b->ndim, true);
    tensor_set_data (expected_b_grad, b->data, b->size);
    tensor_set_grad (expected_b_grad, (float[]) { 5., 6., 7., 8. }, b->size);

    cr_assert (tensor_equals (a, expected_a_grad, true), "cat_axis0_backward a failed");
    cr_assert (tensor_equals (b, expected_b_grad, true), "cat_axis0_backward b failed");
}

Test (cat, cat_axis1_backward)
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4. }, (int[]) { 2, 2 }, 2, true);
    smart tensor_t *b = tensor ((float[]) { 5., 6., 7., 8. }, (int[]) { 2, 2 }, 2, true);
    smart tensor_t *c = tensor_cat ((tensor_t *[]) { a, b }, 2, 1);
    smart tensor_t *weights
        = tensor ((float[]) { 1., 2., 3., 4., 5., 6., 7., 8. }, (int[]) { 2, 4 }, 2, false);
    smart tensor_t *loss = tensor_mul (c, weights);
    smart tensor_t *y    = tensor_sum (loss, NULL, 0);

    tensor_backward (y);

    // c shape [2,4], data: [1,2,5,6, 3,4,7,8]
    // weights: [1,2,3,4, 5,6,7,8]
    // grad_c = weights
    // For a (cols 0-1), grad_a = weights[:,0:2] = [1,2, 5,6]
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 1., 2., 5., 6. }, a->size);

    // For b (cols 2-3), grad_b = weights[:,2:4] = [3,4, 7,8]
    smart tensor_t *expected_b_grad = tensor_create (b->shape, b->ndim, true);
    tensor_set_data (expected_b_grad, b->data, b->size);
    tensor_set_grad (expected_b_grad, (float[]) { 3., 4., 7., 8. }, b->size);

    cr_assert (tensor_equals (a, expected_a_grad, true), "cat_axis1_backward a failed");
    cr_assert (tensor_equals (b, expected_b_grad, true), "cat_axis1_backward b failed");
}

Test (sum, backward_sum_multiple_axes)
{
    log_set_level (LOG_INFO);

    // Test tensor_sum over multiple axes (0 and 1) for a 2D tensor
    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4., 5., 6. }, (int[]) { 2, 3 }, 2, true);
    smart tensor_t *y = tensor_sum (a, (int[]) { 0, 1 }, 2); // sum over both axes

    tensor_backward (y);

    // Expected gradients: dy/da = 1 for each element since sum over all elements
    smart tensor_t *expected_a_grad = tensor_create (a->shape, a->ndim, true);
    tensor_set_data (expected_a_grad, a->data, a->size);
    tensor_set_grad (expected_a_grad, (float[]) { 1., 1., 1., 1., 1., 1. }, a->size);

    cr_assert (tensor_equals (a, expected_a_grad, true), "backward_sum_multiple_axes failed");
}