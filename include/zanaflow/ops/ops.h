#ifndef OPS_H
#define OPS_H

#include <zanaflow/tensor/tensor.h>

int zf_tensor_same_shape(const Tensor *a, const Tensor *b);
Tensor *zf_tensor_add(Tensor *a, Tensor *b);
Tensor *zf_tensor_sub(Tensor *a, Tensor *b);
Tensor *zf_tensor_mul(Tensor *a, Tensor *b);
Tensor *zf_tensor_add_bias(Tensor *a, Tensor *b);
Tensor *zf_tensor_sum_all(Tensor *a);
Tensor *zf_tensor_mean_all(Tensor *a);
Tensor *zf_tensor_mat_mul(const Tensor *a, const Tensor *b);
Tensor *zf_tensor_mat_mul_transpose(const Tensor *a, const Tensor *b);
void    zf_tensor_add_inplace(Tensor *a, const Tensor *b);
Tensor *zf_tensor_sum_rows(const Tensor *a);

#endif
