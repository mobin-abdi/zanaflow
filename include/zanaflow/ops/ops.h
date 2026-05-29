#ifndef OPS_H
#define OPS_H

#include <zanaflow/tensor/tensor.h>

int tensor_same_shape(const Tensor *a, const Tensor *b);

Tensor *tensor_add(const Tensor *a, const Tensor *b);
Tensor *tensor_sub(const Tensor *a, const Tensor *b);
Tensor *tensor_mul(const Tensor *a, const Tensor *b);
// Tensor *tensor_div(const Tensor *a, const Tensor *b);

float tensor_sum_all(const Tensor *a);
float tensor_mean_all(const Tensor *a);

Tensor *tensor_add_bias(const Tensor *a, const Tensor *b);

#endif 
