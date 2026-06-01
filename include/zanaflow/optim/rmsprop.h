#ifndef RMSPROP_H
#define RMSPROP_H

#include <zanaflow/core/parameter.h>

typedef struct
{
    Parameter **params;
    int count;
    float lr;
    float alpha;
    float eps;
    float **cache;
} RMSProp;

RMSProp *zf_rmsprop_create(Parameter **params, int count, float lr, float alpha, float eps);
int zf_rmsprop_zero_grad(RMSProp *opt);
int zf_rmsprop_step(RMSProp *opt);
void zf_rmsprop_free(RMSProp *opt);

#endif
