#include "log.h"

#define ASSERT(condition, format, ...)        \
    do                                        \
    {                                         \
        if (!(condition))                     \
        {                                     \
            log_error(format, ##__VA_ARGS__); \
            exit(EXIT_FAILURE);               \
        }                                     \
    } while (0)
