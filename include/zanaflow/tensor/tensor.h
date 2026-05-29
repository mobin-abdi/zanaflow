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

Tensor *tensor_create(int *shape, int ndim);
void tensor_retain(Tensor *t);
void tensor_release(Tensor *t);
int tensor_ensure_grad(Tensor *t);
void tensor_zero_grad(Tensor *t);
void tensor_print(Tensor *t);
Tensor *tensor_clone(Tensor *t);

#endif
