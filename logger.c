#include "logger.h"

void log_error(const char* message) {
    if (log_level >= LOG_LEVEL_ERROR) {
        printf("ERROR: %s\n", message);
    }
}

void log_warn(const char* message) {
    if (log_level >= LOG_LEVEL_WARN) {
        printf("WARN: %s\n", message);
    }
}

void log_info(const char* message) {
    if (log_level >= LOG_LEVEL_INFO) {
        printf("INFO: %s\n", message);
    }
}

void log_debug(const char* message) {
    if (log_level >= LOG_LEVEL_DEBUG) {
        printf("DEBUG: %s\n", message);
    }
}