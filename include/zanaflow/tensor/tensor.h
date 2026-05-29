#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float *data;
    int *shape;
    int ndim;
    int size;
} Tensor;

int compute_size(int *shape, int ndim);
Tensor *tensor_create(int *shape, int ndim);
void tensor_fill(Tensor *tensor, float value);
void tensor_print(Tensor *t);
void tensor_free(Tensor *t);
Tensor *tensor_clone(const Tensor *t);
Tensor *tensor_mul_scalar(const Tensor *a, float s);
Tensor *tensor_add_scalar(const Tensor *a, float s);
Tensor *tensor_zeros(int *shape, int ndim);
Tensor *tensor_ones(int *shape, int ndim);
Tensor *tensor_flatten(const Tensor *t);
Tensor *tensor_reshape(const Tensor *t, int *new_shape, int new_ndim);

#endif 