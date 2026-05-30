#ifndef ZANAFLOW_NN_DENSE_H
#define ZANAFLOW_NN_DENSE_H

#include <zanaflow/tensor/tensor.h>
#include <zanaflow/core/parameter.h> 

typedef struct {
    Tensor *weights;
    Tensor *bias;
    Tensor *grad_weights;
    Tensor *grad_bias;
    Tensor *last_input;
    int in_features;
    int out_features;
} DenseLayer;

DenseLayer *zf_dense_create(int in_features, int out_features);
Tensor *zf_dense_forward(DenseLayer *layer, const Tensor *input);
Tensor *zf_dense_backward(DenseLayer *layer, const Tensor *grad_output);
int zf_dense_parameters(DenseLayer *layer, Parameter out_params[2]);
void zf_dense_free(DenseLayer *layer);

#endif 
