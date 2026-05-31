#include <stdlib.h>
#include <zanaflow/nn/init.h>
#include <zanaflow/nn/dense.h>
#include <zanaflow/ops/ops.h>
#include <stdio.h>

DenseLayer *zf_dense_create(int in_features, int out_features)
{
    DenseLayer *layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    if (!layer)
    {
        return NULL;
    }

    layer->in_features = in_features;
    layer->out_features = out_features;

    int w_shape[2] = {in_features, out_features};
    int b_shape[2] = {1, out_features};

    layer->weights = zf_parameter_create(w_shape, 2);
    layer->bias = zf_parameter_create(b_shape, 2);

    if (!layer->weights || !layer->bias)
    {
        if (layer->weights)
            zf_parameter_free(layer->weights);
        if (layer->bias)
            zf_parameter_free(layer->bias);
        free(layer);
        return NULL;
    }

    return layer;
}

Tensor *zf_dense_forward(DenseLayer *layer, Tensor *input)
{
    if (!layer || !input)
    {
        return NULL;
    }
    
    if (!layer->weights || !layer->bias || !layer->weights->value || !layer->bias->value)
    {
        return NULL;
    }

    Tensor *z = zf_tensor_mat_mul(input, layer->weights->value);
    if (!z) return NULL;

    Tensor *y = zf_tensor_add_bias(z, layer->bias->value);
    zf_tensor_release(z);
    return y;
}

int zf_dense_parameters(DenseLayer *layer, Parameter *out_params[2])
{
    if (!layer || !out_params)
    {
        return 0;
    }
    out_params[0] = layer->weights;
    out_params[1] = layer->bias;
    return 2;
}

void zf_dense_free(DenseLayer *layer)
{
    if (!layer)
    {
        return;
    }

    if (layer->weights)
    {
        zf_parameter_free(layer->weights);
    }

    if (layer->bias)
    {
        zf_parameter_free(layer->bias);
    }

    free(layer);
}