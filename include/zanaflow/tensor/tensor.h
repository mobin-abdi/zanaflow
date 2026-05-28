#ifndef TENSOR_H
#define TENDOR_H

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

#endif 