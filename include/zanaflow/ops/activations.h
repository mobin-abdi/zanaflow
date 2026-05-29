#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <zanaflow/tensor/tensor.h>

Tensor *relu_activation(const Tensor *a);
Tensor *sigmoid_activation(const Tensor *a);
Tensor *leaky_relu_activation(const Tensor *a, float alpha);
Tensor *tanh_activation(const Tensor *a);
Tensor *softmax_activation(const Tensor *a);

#endif 
