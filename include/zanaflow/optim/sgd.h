#ifndef ZANAFLOW_OPTIM_OPTIM_H
#define ZANAFLOW_OPTIM_OPTIM_H

#include <zanaflow/core/parameter.h>

typedef struct {
    Parameter *params;
    int count;
    float lr;
} SGD;

SGD *zf_sgd_create(Parameter *params, int count, float lr);
void zf_sgd_zero_grad(SGD *opt);
void zf_sgd_step(SGD *opt);
void zf_sgd_free(SGD *opt);

#endif
