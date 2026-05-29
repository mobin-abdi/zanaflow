#include <math.h>
#include <stdlib.h>
#include <zanaflow/ops/activations.h>

Tensor *relu_activation(const Tensor *a) {
    if (a == NULL) return NULL;

    Tensor *out = tensor_create(a->shape, a->ndim);
    if (out == NULL) return NULL;

    for (int i = 0; i < a->size; i++) {
        // f(x) = max(0, x)
        out->data[i] = (a->data[i] > 0.0f) ? a->data[i] : 0.0f;
    }

    return out;
}

Tensor *sigmoid_activation(const Tensor *a) {
    if (a == NULL) return NULL;

    Tensor *out = tensor_create(a->shape, a->ndim);
    if (out == NULL) return NULL;

    for (int i = 0; i < a->size; i++) {
        // f(x) = 1 / (1 + exp(-x))
        out->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
    }

    return out;
}

Tensor *leaky_relu_activation(const Tensor *a, float alpha)
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
        float x = a -> data[i];
        out -> data[i] = (x > 0.0f) ? x : (alpha * x);
    }

    return out;
}
Tensor *tanh_activation(const Tensor *a)
{
    if (a == NULL)
    {
        return NULL;
    }

    Tensor *out = tensor_create(a -> shape, a -> ndim);
    if (a == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < a -> size; i++)
    {
        out -> data[i] = tanhf(a -> data[i]);        
    }
    return out;
}

Tensor *softmax_activation(const Tensor *a)
{
    if (a == NULL)
    {
        return NULL;
    }

    if (a->ndim != 1) 
    {
        return NULL;
    }

    if (a->size <= 0)
    {
        return NULL;
    }

    Tensor *out = tensor_create(a -> shape, a -> ndim);

    if (out == NULL)
    {
        return NULL;
    }

    float m = a->data[0];
    for (int i = 1; i < a->size; i++) {
        if (a->data[i] > m) m = a->data[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < a->size; i++) {
        float e = expf(a->data[i] - m);
        out->data[i] = e;
        sum += e;
    }

    if (sum == 0.0f)
    {
        tensor_free(out);
        return NULL;
    }

    for (int i = 0; i < a->size; i++) {
        out->data[i] /= sum;
    }

    return out;
}

