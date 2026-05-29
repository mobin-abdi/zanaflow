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

Tensor *tensor_clone(const Tensor *t)
{
    if (t == NULL)
    {
        return NULL;
    }

    Tensor *out = tensor_create(t -> shape, t -> ndim);

    if (out == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < t -> size; i++)
    {
        out -> data[i] = t -> data[i];
    }

    return out;
}

Tensor *tensor_mul_scalar(const Tensor *a, float s)
{
    if (a == NULL)
    {
        return NULL;
    }

    Tensor *out = tensor_create(a -> shape, a -> ndim);

    if (out == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < a -> size; i++)
    {
        out -> data[i] = a -> data[i] * s;
    }

    return out;
}

Tensor *tensor_add_scalar(const Tensor *a, float s)
{
    if (a == NULL)
    {
        return NULL;
    }

    Tensor *out = tensor_create(a -> shape, a -> ndim);

    if (out == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < a -> size; i++)
    {
        out -> data[i] = a -> data[i] + s;
    }

    return out;
}

Tensor *tensor_zeros(int *shape, int ndim)
{
    Tensor *t = tensor_create(shape, ndim);
    if (t == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < t -> size; i++)
    {
        t -> data[i] = 0.0f;
    }

    return t;
}

Tensor *tensor_ones(int *shape, int ndim)
{
    Tensor *t = tensor_create(shape, ndim);
    if (t == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < t -> size; i++)
    {
        t -> data[i] = 1.0f;
    }

    return t;
}

Tensor *tensor_flatten(const Tensor *t)
{
    if (!t)
    {
        return NULL;
    }

    int new_shape[1] = { t->size };
    Tensor *out = tensor_create(new_shape, 1);
    if (!out)
    {
        return NULL;
    }

    for (int i = 0; i < t->size; i++)
        out->data[i] = t->data[i];

    return out;
}


Tensor *tensor_reshape(const Tensor *t, int *new_shape, int new_ndim)
{
    if (t == NULL) return NULL;

    int new_size = 1;
    for (int i = 0; i < new_ndim; i++)
    {
        new_size *= new_shape[i];
    }

    if (new_size != t->size)
    {
        return NULL;
    }

    Tensor *out = tensor_create(new_shape, new_ndim);
    if (out == NULL) return NULL;

    for (int i = 0; i < t->size; i++)
    {
        out->data[i] = t->data[i];
    }

    return out;
}

void tensor_fill_random_uniform(Tensor *tensor, float low, float high)
{
    float range = high - low;
    for (int i = 0; i < tensor->size; i++)
    {
        tensor->data[i] = low + ((float)rand() / (float)RAND_MAX) * range;
    }
}