#include <zanaflow/nn/dense.h>
#include <zanaflow/nn/init.h>
#include <zanaflow/ops/ops.h>
#include <stdlib.h>

DenseLayer *zf_dense_create(int in_features, int out_features)
{
    if (in_features <= 0 || out_features <= 0)
    {
        return NULL;
    }

    DenseLayer *layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    if (!layer)
    {
        return NULL;
    }

    int w_shape[] = {in_features, out_features};
    layer->weights = zf_tensor_create(w_shape, 2);

    int b_shape[] = {out_features};
    layer->bias = zf_tensor_create(b_shape, 1);

    if (!layer->weights || !layer->bias)
    {
        if (layer->weights)
        {
            zf_tensor_release(layer->weights);
        }
        if (layer->bias)
        {
            zf_tensor_release(layer->bias);
        }
        free(layer);
        return NULL;
    }

    zf_init_he_uniform(layer->weights, layer->bias, in_features);

    if (!zf_tensor_ensure_grad(layer->weights) || !zf_tensor_ensure_grad(layer->bias))
    {
        zf_tensor_release(layer->weights);
        zf_tensor_release(layer->bias);
        free(layer);
        return NULL;
    }

    zf_tensor_zero_grad(layer->weights);
    zf_tensor_zero_grad(layer->bias);

    layer->last_input = NULL;
    layer->in_features = in_features;
    layer->out_features = out_features;

    return layer;
}

Tensor *zf_dense_forward(DenseLayer *layer, Tensor *input)
{
    if (!layer || !input)
    {
        return NULL;
    }

    if (input->ndim < 2 || input->shape[1] != layer->in_features)
    {
        return NULL;
    }

    if (layer->last_input)
    {
        zf_tensor_release(layer->last_input);
    }

    layer->last_input = zf_tensor_clone(input);
    if (!layer->last_input)
    {
        return NULL;
    }

    Tensor *output_intermediate = zf_tensor_mat_mul(input, layer->weights);
    if (!output_intermediate)
    {
        return NULL;
    }

    Tensor *output = zf_tensor_add_bias(output_intermediate, layer->bias);
    zf_tensor_release(output_intermediate);

    return output;
}

Tensor *zf_dense_backward(DenseLayer *layer, const Tensor *grad_output)
{
    if (!layer || !grad_output || !layer->last_input)
    {
        return NULL;
    }

    if (!zf_tensor_ensure_grad(layer->weights) || !zf_tensor_ensure_grad(layer->bias))
    {
        return NULL;
    }

    Tensor *dW = zf_tensor_mat_mul_transpose(layer->last_input, grad_output);
    if (!dW)
    {
        return NULL;
    }

    for (int i = 0; i < dW->size; i++)
    {
        layer->weights->grad[i] += dW->data[i];
    }
    zf_tensor_release(dW);

    Tensor *db = zf_tensor_sum_rows(grad_output);
    if (!db)
    {
        return NULL;
    }

    for (int i = 0; i < db->size; i++)
    {
        layer->bias->grad[i] += db->data[i];
    }
    zf_tensor_release(db);

    Tensor *dX = zf_tensor_mat_mul_transpose(grad_output, layer->weights);
    return dX;
}

int zf_dense_parameters(DenseLayer *layer, Parameter out_params[2])
{
    if (!layer || !out_params)
    {
        return 0;
    }

    out_params[0].value = layer->weights;
    out_params[1].value = layer->bias;

    return 2;
}

void zf_dense_free(DenseLayer *layer)
{
    if (!layer)
    {
        return;
    }

    zf_tensor_release(layer->weights);
    zf_tensor_release(layer->bias);

    if (layer->last_input)
    {
        zf_tensor_release(layer->last_input);
    }

    free(layer);
}
