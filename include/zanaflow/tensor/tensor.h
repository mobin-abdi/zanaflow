#ifndef TENSOR_H
#define TENSOR_H

#include <zanaflow/core/refcount.h>

typedef struct AutogradNode AutogradNode;

typedef struct Tensor {
    float *data;
    float *grad;
    int *shape;
    int ndim;
    int size;
    int requires_grad;
    AutogradNode *grad_node;
    RefCount ref;
} Tensor;

Tensor *zf_tensor_create(int *shape, int ndim);
void zf_tensor_retain(Tensor *t);
void zf_tensor_release(Tensor *t);
int zf_tensor_ensure_grad(Tensor *t);
void zf_tensor_zero_grad(Tensor *t);
void zf_tensor_print(Tensor *t);
Tensor *zf_tensor_clone(const Tensor *t);
void zf_tensor_fill(Tensor *t, float value);
Tensor *zf_tensor_full(const int *shape, int ndim, float value);
Tensor *zf_tensor_zeros(const int *shape, int ndim);
Tensor *zf_tensor_ones(const int *shape, int ndim);
Tensor *zf_tensor_reshape(const Tensor *t, const int *new_shape, int new_ndim);

#endif
