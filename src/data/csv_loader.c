#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <zanaflow/data/csv_loader.h>

#ifndef ZF_CSV_LINE_BUF_SIZE
#define ZF_CSV_LINE_BUF_SIZE 4096
#endif

static int zf_is_blank_line(const char *s)
{
    while (*s)
    {
        if (!isspace((unsigned char)*s))
        {
            return 0;
        }
        s++;
    }
    return 1;
}

static char *zf_ltrim(char *s)
{
    while (*s && isspace((unsigned char)*s))
    {
        s++;
    }
    return s;
}

static int zf_parse_csv_line_f32(char *line, size_t expected_cols, float *vals_out)
{
    size_t col = 0;
    char *p = line;

    while (col < expected_cols)
    {
        p = zf_ltrim(p);

        if (*p == '\0' || *p == '\n' || *p == '\r')
        {
            return -1;
        }

        errno = 0;
        char *endptr = NULL;
        double v = strtod(p, &endptr);

        if (endptr == p || errno != 0)
        {
            return -2;
        }

        vals_out[col] = (float)v;
        col++;

        p = endptr;
        p = zf_ltrim(p);

        if (col == expected_cols)
        {
            break;
        }

        if (*p != ',')
        {
            return -3;
        }
        p++;
    }

    return 0;
}

void zf_csv_data_free(zf_csv_data_f32 *data)
{
    if (!data)
    {
        return;
    }
    free(data->X);
    free(data->y);
    data->X = NULL;
    data->y = NULL;
    data->n = 0;
    data->d = 0;
}

int zf_csv_load_f32(
    const char *path,
    size_t feature_dim,
    int has_header,
    int label_in_last_column,
    zf_csv_data_f32 *out)
{
    if (!path || !out)
    {
        fprintf(stderr, "zf_csv_load_f32: null argument\n");
        return -1;
    }

    out->X = NULL;
    out->y = NULL;
    out->n = 0;
    out->d = 0;

    if (feature_dim == 0)
    {
        fprintf(stderr, "zf_csv_load_f32: feature_dim must be > 0\n");
        return -2;
    }

    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        fprintf(stderr, "zf_csv_load_f32: failed to open '%s'\n", path);
        return -3;
    }

    const size_t expected_cols = feature_dim + 1;
    char line[ZF_CSV_LINE_BUF_SIZE];

    if (has_header)
    {
        if (!fgets(line, sizeof(line), fp))
        {
            fprintf(stderr, "zf_csv_load_f32: failed to read header\n");
            fclose(fp);
            return -4;
        }
    }

    size_t n = 0;
    while (fgets(line, sizeof(line), fp))
    {
        char *s = zf_ltrim(line);
        if (*s == '\0' || *s == '\n' || *s == '\r')
        {
            continue;
        }
        if (zf_is_blank_line(s))
        {
            continue;
        }
        if (*s == '#')
        {
            continue;
        }
        n++;
    }

    if (n == 0)
    {
        fprintf(stderr, "zf_csv_load_f32: no data rows in '%s'\n", path);
        fclose(fp);
        return -5;
    }

    float *X = (float *)malloc(n * feature_dim * sizeof(float));
    float *y = (float *)malloc(n * sizeof(float));
    float *tmp = (float *)malloc(expected_cols * sizeof(float));

    if (!X || !y || !tmp)
    {
        fprintf(stderr, "zf_csv_load_f32: out of memory\n");
        free(X);
        free(y);
        free(tmp);
        fclose(fp);
        return -6;
    }

    rewind(fp);
    if (has_header)
    {
        if (!fgets(line, sizeof(line), fp))
        {
            fprintf(stderr, "zf_csv_load_f32: failed to reread header\n");
            free(X);
            free(y);
            free(tmp);
            fclose(fp);
            return -7;
        }
    }

    size_t row = 0;
    while (fgets(line, sizeof(line), fp))
    {
        char *s = zf_ltrim(line);
        if (*s == '\0' || *s == '\n' || *s == '\r')
        {
            continue;
        }
        if (zf_is_blank_line(s))
        {
            continue;
        }
        if (*s == '#')
        {
            continue;
        }

        int prc = zf_parse_csv_line_f32(s, expected_cols, tmp);
        if (prc != 0)
        {
            fprintf(stderr, "zf_csv_load_f32: parse error at data row %zu (code %d)\n", row, prc);
            free(X);
            free(y);
            free(tmp);
            fclose(fp);
            return -8;
        }

        if (label_in_last_column)
        {
            for (size_t j = 0; j < feature_dim; j++)
            {
                X[row * feature_dim + j] = tmp[j];
            }
            y[row] = tmp[feature_dim];
        }
        else
        {
            y[row] = tmp[0];
            for (size_t j = 0; j < feature_dim; j++)
            {
                X[row * feature_dim + j] = tmp[j + 1];
            }
        }

        row++;
    }

    fclose(fp);
    free(tmp);

    if (row != n)
    {
        fprintf(stderr, "zf_csv_load_f32: internal row count mismatch (%zu vs %zu)\n", row, n);
        free(X);
        free(y);
        return -9;
    }

    out->X = X;
    out->y = y;
    out->n = n;
    out->d = feature_dim;

    return 0;
}
