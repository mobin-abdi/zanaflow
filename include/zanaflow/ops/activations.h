#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <zanaflow/tensor/tensor.h>

Tensor *zf_relu(const Tensor *a);
Tensor *zf_sigmoid(const Tensor *a);
Tensor *zf_leaky_relu(const Tensor *a, float alpha);
Tensor *zf_tanh(const Tensor *a);
Tensor *zf_softmax(const Tensor *a);

#endif 
