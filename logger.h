#include <stdio.h>

#define LOG_LEVEL_NONE 0
#define LOG_LEVEL_ERROR 1
#define LOG_LEVEL_WARN 2
#define LOG_LEVEL_INFO 3
#define LOG_LEVEL_DEBUG 4

extern int log_level;

void log_error(const char* message);
void log_warn(const char* message);
void log_info(const char* message);
void log_debug(const char* message);