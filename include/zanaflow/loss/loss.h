#ifndef LOSS_H
#define LOSS_H

#include <zanaflow/tensor/tensor.h>

float mse_loss(const Tensor *pred, const Tensor *target);

#endif