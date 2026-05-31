#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <zanaflow/tensor/tensor.h>

static void zf_relu_backward(AutogradNode *node, float *grad_output);
static void zf_sigmoid_backward(AutogradNode *node, float *grad_output);
static void zf_tanh_backward(AutogradNode *node, float *grad_output);
static void zf_leaky_relu_backward(AutogradNode *node, float *grad_output);
static void zf_softmax_backward(AutogradNode *node, float *grad_output);

Tensor *zf_relu(const Tensor *a);
Tensor *zf_sigmoid(const Tensor *a);
Tensor *zf_leaky_relu(const Tensor *a, float alpha);
Tensor *zf_tanh(const Tensor *a);
Tensor *zf_softmax(const Tensor *a);

#endif 
