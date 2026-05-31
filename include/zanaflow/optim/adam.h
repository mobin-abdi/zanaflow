#ifndef ADAM_H
#define ADAM_H

#include <zanaflow/core/parameter.h>

typedef struct
{
    Parameter **params;
    int count;
    float lr;
    float beta1;
    float beta2;
    float eps;
    int t;
    float **m;
    float **v;
} Adam;

Adam *zf_adam_create(Parameter **params, int count, float lr, float beta1, float beta2, float eps);
void zf_adam_zero_grad(Adam *opt);
void zf_adam_step(Adam *opt);
void zf_adam_free(Adam *opt);

#endif