#include <stdio.h>
#include <stdlib.h>
#include <zanaflow/tensor/tensor.h>

int compute_size(int *shape, int ndim)
{
    int size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size *= shape[i];
    }
    return size;
}

Tensor *tensor_create(int *shape, int ndim)
{
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (t == NULL)
    {
        return NULL;
    }

    t->ndim = ndim;
    t->size = compute_size(shape, ndim);

    t->shape = (int *)malloc(ndim * sizeof(int));
    if (t->shape == NULL)
    {
        free(t);
        return NULL;
    }

    for (int i = 0; i < ndim; i++)
    {
        t->shape[i] = shape[i];
    }

    t->data = (float *)malloc(t->size * sizeof(float));
    if (t->data == NULL)
    {
        free(t->shape);
        free(t);
        return NULL;
    }

    return t;
}

void tensor_fill(Tensor *tensor, float value)
{
    for (int i = 0; i < tensor->size; i++)
    {
        tensor->data[i] = value;
    }
}

void tensor_print(Tensor *t)
{
    printf("Tensor(shape=[");
    for (int i = 0; i < t->ndim; i++)
    {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("], size=%d)\n", t->size);

    printf("data = [");
    for (int i = 0; i < t->size; i++)
    {
        printf("%.2f", t->data[i]);
        if (i < t->size - 1) printf(", ");
    }
    printf("]\n");
}

void tensor_free(Tensor *t)
{
    if (!t) return;
    free(t->data);
    free(t->shape);
    free(t);
}
