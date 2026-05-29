#ifndef OPS_H
#define OPS_H

#include <zanaflow/tensor/tensor.h>

int tensor_same_shape(const Tensor *a, const Tensor *b);
Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_sub(Tensor *a, Tensor *b);
Tensor *tensor_mul(Tensor *a, Tensor *b);
Tensor *tensor_add_bias(Tensor *a, Tensor *b);
Tensor *tensor_sum_all(Tensor *a);
Tensor *tensor_mean_all(Tensor *a);

#endif 
