#include <criterion/criterion.h>

// a simple test
Test(my_test, my_first_test)
{
    cr_assert(1 == 1, "1 is not 1...?");
}