#ifndef PARAMETER_H
#define PARAMETER_H

#include <zanaflow/tensor/tensor.h>

typedef struct
{
    Tensor *value;
} Parameter;

Parameter *zf_parameter_create(int *shape, int ndim);
void zf_parameter_free(Parameter *p);

#endif
