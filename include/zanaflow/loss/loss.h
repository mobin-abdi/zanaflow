#ifndef LOSS_H
#define LOSS_H

#include <zanaflow/tensor/tensor.h>

Tensor* zf_loss_mse(Tensor *pred, Tensor *target);
Tensor *zf_loss_cross_entropy(Tensor *logits, Tensor *target_indices);
Tensor *zf_loss_binary_cross_entropy(Tensor *logits, Tensor *targets);

#endif
