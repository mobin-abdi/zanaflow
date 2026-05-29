#ifndef ZANAFLOW_OPTIM_OPTIM_H
#define ZANAFLOW_OPTIM_OPTIM_H

#include <zanaflow/core/parameter.h>

typedef struct {
    Parameter *params;
    int count;
    float lr;
} SGD;

SGD *sgd_create(Parameter *params, int count, float lr);
void sgd_zero_grad(SGD *opt);
void sgd_step(SGD *opt);
void sgd_free(SGD *opt);

#endif
