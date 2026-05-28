#include <stdlib.h>
#include <zanaflow/ops/ops.h>

int tensor_same_shape(const Tensor *a, const Tensor *b)
{
    if (a == NULL)
    {
        return 0;
    }

    if (b == NULL)
    {
        return 0;
    }

    if (a->ndim != b->ndim)
    {
        return 0;
    }

    for (int i = 0; i < a->ndim; i++)
    {
        if (a->shape[i] != b->shape[i])
        {
            return 0;
        }
    }

    return 1;
}

Tensor *tensor_add(const Tensor *a, const Tensor *b)
{
    if (!tensor_same_shape(a, b))
    {
        return NULL;
    }

    Tensor *out = tensor_create(a->shape, a->ndim);
    if (out == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }

    return out;
}

Tensor *tensor_sub(const Tensor *a, const Tensor *b)
{
    if (!tensor_same_shape(a, b))
    {
        return NULL;
    }

    Tensor *out = tensor_create(a->shape, a->ndim);
    if (out == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = a->data[i] - b->data[i];
    }

    return out;
}

Tensor *tensor_mul(const Tensor *a, const Tensor *b)
{
    if (!tensor_same_shape(a, b))
    {
        return NULL;
    }

    Tensor *out = tensor_create(a->shape, a->ndim);
    if (out == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = a->data[i] * b->data[i];
    }

    return out;
}

float tensor_sum_all(const Tensor *a)
{
    if (a == NULL)
    {
        return 0.0f;
    }

    float sum = 0.0f;

    for (int i = 0; i < a->size; i++)
    {
        sum += a->data[i];
    }

    return sum;
}

float tensor_mean_all(const Tensor *a)
{
    if (a == NULL || a->size == 0)
    {
        return 0.0f;
    }

    return tensor_sum_all(a) / (float)a->size;
}
