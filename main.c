#include "forward.h"
#include "log.h"
#include "tensor.h"

int
main ()
{
    log_set_level (LOG_INFO);

    // Current bug illustration
    smart tensor_t *tout = tensor_zeros ((int[]) { 1 }, 1, false);

    // 1d example
    // smart tensor_t *tin  = tensor ((float[]) { 1., 2., 3. }, (int[]) { 3 }, 1, true);
    // smart tensor_t *col0 = tensor_slice (tin, (slice_t[]) { SLICE_ONE (0) });
    // smart tensor_t *col1 = tensor_slice (tin, (slice_t[]) { SLICE_ONE (1) });
    // smart tensor_t *col2 = tensor_slice (tin, (slice_t[]) { SLICE_ONE (2) });

    // TENSOR_REBIND (tout, tensor_add (tout, col0));
    // TENSOR_REBIND (tout, tensor_add (tout, col1));
    // TENSOR_REBIND (tout, tensor_add (tout, col2));

    // 2d example

    // sum axis 0
    smart tensor_t *trow = tensor_zeros ((int[]) { 3 }, 1, false);
    smart tensor_t *tin  = tensor ((float[]) { 1., 2., 3., 4., 5., 6. }, (int[]) { 2, 3 }, 2, true);

    smart tensor_t *row0 = tensor_slice (tin, (slice_t[]) { SLICE_ONE (0), SLICE_ALL });
    TENSOR_REBIND (row0, tensor_reshape (row0, (int[]) { 3 }, 1));

    smart tensor_t *row1 = tensor_slice (tin, (slice_t[]) { SLICE_ONE (1), SLICE_ALL });
    TENSOR_REBIND (row1, tensor_reshape (row1, (int[]) { 3 }, 1));

    TENSOR_REBIND (trow, tensor_add (trow, row0));
    TENSOR_REBIND (trow, tensor_add (trow, row1));

    smart tensor_t *col0 = tensor_slice (trow, (slice_t[]) { SLICE_ONE (0) });
    smart tensor_t *col1 = tensor_slice (trow, (slice_t[]) { SLICE_ONE (1) });
    smart tensor_t *col2 = tensor_slice (trow, (slice_t[]) { SLICE_ONE (2) });

    TENSOR_REBIND (tout, tensor_add (tout, col0));
    TENSOR_REBIND (tout, tensor_add (tout, col1));
    TENSOR_REBIND (tout, tensor_add (tout, col2));

    tensor_backward (tout);

    printf ("IN\n");
    tensor_print (tin, PRINT_ALL);

    printf ("OUT\n");
    tensor_print (tout, PRINT_ALL);

    printf ("SUM ROW\n");
    tensor_print (trow, PRINT_ALL);

    printf ("ROW 0\n");
    tensor_print (row0, PRINT_ALL);

    printf ("ROW 1\n");
    tensor_print (row1, PRINT_ALL);

    printf ("COL 0\n");
    tensor_print (col0, PRINT_ALL);

    printf ("COL 1\n");
    tensor_print (col1, PRINT_ALL);

    printf ("COL 2\n");
    tensor_print (col2, PRINT_ALL);
}