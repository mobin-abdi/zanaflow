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

/**
 * @brief Creates and initializes a dense layer.
 * Allocates weights, bias, and their gradients.
 * @param in_features Number of input features.
 * @param out_features Number of output features.
 * @return Pointer to the allocated DenseLayer structure, or NULL on failure.
 */
DenseLayer *dense_create(int in_features, int out_features);

/**
 * @brief Performs forward propagation: Output = Input * Weights + Bias.
 * Caches the input for the backward pass.
 * @param layer Pointer to the dense layer.
 * @param input Input tensor. Must have shape [batch_size, in_features].
 * @return Pointer to the output tensor. Shape will be [batch_size, out_features].
 *         Caller is responsible for freeing the returned tensor.
 */
Tensor *dense_forward(DenseLayer *layer, const Tensor *input);

/**
 * @brief Performs backward propagation to compute gradients.
 * Calculates gradients for weights and bias, and returns the gradient w.r.t. input.
 * @param layer Pointer to the dense layer.
 * @param grad_output Gradient of the loss w.r.t. the output of this layer.
 *                    Shape must match the output of dense_forward.
 * @return Gradient of the loss w.r.t. the input tensor (for the previous layer).
 *         Caller is responsible for freeing the returned tensor.
 */
Tensor *dense_backward(DenseLayer *layer, const Tensor *grad_output);

/**
 * @brief Extracts the trainable parameters (weights and bias) of the dense layer.
 * @param layer Pointer to the dense layer.
 * @param out_params An array of size 2 to store the parameters.
 *                   out_params[0] will be weights, out_params[1] will be bias.
 * @return The number of parameters extracted (always 2 for DenseLayer). Returns 0 on error.
 */
int dense_parameters(DenseLayer *layer, Parameter out_params[2]);

/**
 * @brief Frees all memory allocated for the dense layer and its tensors.
 * @param layer Pointer to the dense layer to free.
 */
void dense_free(DenseLayer *layer);

#endif 

