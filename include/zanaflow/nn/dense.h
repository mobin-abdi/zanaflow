#ifndef ZANAFLOW_NN_DENSE_H
#define ZANAFLOW_NN_DENSE_H

#include <zanaflow/tensor/tensor.h>
#include <zanaflow/core/parameter.h>

typedef struct
{
    int in_features;
    int out_features;
    Parameter *weights;
    Parameter *bias;
} DenseLayer;

DenseLayer *zf_dense_create(int in_features, int out_features);
Tensor *zf_dense_forward(DenseLayer *layer, Tensor *input);
int zf_dense_parameters(DenseLayer *layer, Parameter *out_params[2]);
void zf_dense_free(DenseLayer *layer);

#endif
