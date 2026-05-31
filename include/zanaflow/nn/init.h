#ifndef INIT_H
#define INIT_H

#include <zanaflow/tensor/tensor.h>

void zf_init_he_uniform(Tensor *weights, Tensor *bias, int fan_in);
void zf_init_xavier_uniform(Tensor *weights, Tensor *bias, int fan_in, int fan_out);

#endif
