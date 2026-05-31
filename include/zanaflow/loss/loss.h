#ifndef LOSS_H
#define LOSS_H

#include <zanaflow/tensor/tensor.h>

Tensor* zf_loss_mse(Tensor *pred, Tensor *target);

#endif
