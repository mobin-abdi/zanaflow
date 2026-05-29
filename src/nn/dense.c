#include <zanaflow/nn/dense.h>
#include <stdlib.h>
#include <string.h> 

DenseLayer *dense_create(int in_features, int out_features) {
    if (in_features <= 0 || out_features <= 0) {
        return NULL;
    }

    DenseLayer *layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    if (!layer) return NULL;

    layer->weights = tensor_create(in_features, out_features);
    layer->bias = tensor_create(1, out_features);
    layer->grad_weights = tensor_create(in_features, out_features);
    layer->grad_bias = tensor_create(1, out_features);

    if (!layer->weights || !layer->bias || !layer->grad_weights || !layer->grad_bias) {
        tensor_free(layer->weights);
        tensor_free(layer->bias);
        tensor_free(layer->grad_weights);
        tensor_free(layer->grad_bias);
        free(layer);
        return NULL;
    }

    tensor_fill(layer->grad_weights, 0.0f);
    tensor_fill(layer->grad_bias, 0.0f);

    layer->last_input = NULL; 
    layer->in_features = in_features;
    layer->out_features = out_features;

    return layer;
}

Tensor *dense_forward(DenseLayer *layer, const Tensor *input) {
    if (!layer || !input) return NULL;

    if (input->cols != layer->in_features) {
        return NULL;
    }

    if (layer->last_input) {
        tensor_free(layer->last_input);
    }

    layer->last_input = tensor_copy(input);
    if (!layer->last_input) return NULL;

    Tensor *output_intermediate = tensor_mat_mul(input, layer->weights);
    if (!output_intermediate) return NULL; // Allocation failed

    Tensor *output = tensor_add(output_intermediate, layer->bias);
    tensor_free(output_intermediate); 
    return output;
}

Tensor *dense_backward(DenseLayer *layer, const Tensor *grad_output) {
    if (!layer || !grad_output || !layer->last_input) {
        return NULL; 
    }
    
    Tensor *dW = tensor_mat_mul_transpose(layer->last_input, grad_output);
    if (!dW) return NULL;

    tensor_add_inplace(layer->grad_weights, dW);
    tensor_free(dW); 

    Tensor *db = tensor_sum_rows(grad_output);
    if (!db) return NULL;

    tensor_add_inplace(layer->grad_bias, db);
    tensor_free(db);
    Tensor *dX = tensor_mat_mul_transpose(grad_output, layer->weights);
    return dX;
}

int dense_parameters(DenseLayer *layer, Parameter out_params[2]) {
    if (!layer || !out_params) return 0;

    out_params[0].value = layer->weights;
    out_params[0].grad  = layer->grad_weights;

    out_params[1].value = layer->bias;
    out_params[1].grad  = layer->grad_bias;

    return 2;
}

void dense_free(DenseLayer *layer) {
    if (!layer) return;

    tensor_free(layer->weights);
    tensor_free(layer->bias);
    tensor_free(layer->grad_weights);
    tensor_free(layer->grad_bias);
    tensor_free(layer->last_input); 

    free(layer);
}
