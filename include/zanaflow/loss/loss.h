#ifndef LOSS_H
#define LOSS_H

#include <zanaflow/tensor/tensor.h>

float zf_loss_mse(const Tensor *pred, const Tensor *target);

#endif