#ifndef CSV_LOADER_H
#define CSV_LOADER_H

#include <stddef.h>

typedef struct {
    float *X;
    float *y;
    size_t n;
    size_t d;
} zf_csv_data_f32;

int zf_csv_load_f32(
    const char *path,
    size_t feature_dim,
    int has_header,
    int label_in_last_column,
    zf_csv_data_f32 *out
);

void zf_csv_data_free(zf_csv_data_f32 *data);

#endif
