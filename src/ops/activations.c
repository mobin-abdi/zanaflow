#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <zanaflow/ops/activations.h>

Tensor *zf_relu(const Tensor *a) {
    if (a == NULL)
    {
        return NULL;
    }

    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (out == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < a->size; i++) 
    {
        out->data[i] = (a->data[i] > 0.0f) ? a->data[i] : 0.0f;
    }

    return out;
}

Tensor *zf_sigmoid(const Tensor *a) 
{
    if (a == NULL)
    {
        return NULL;
    }

    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (out == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < a->size; i++) 
    {
        out->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
    }

    return out;
}

Tensor *zf_leaky_relu(const Tensor *a, float alpha)
{
    if (a == NULL)
    {
        return NULL;
    }

    Tensor *out = zf_tensor_create(a -> shape, a -> ndim);
    
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
Tensor *zf_tanh(const Tensor *a)
{
    if (a == NULL)
    {
        return NULL;
    }

    Tensor *out = zf_tensor_create(a -> shape, a -> ndim);
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

Tensor *zf_softmax(const Tensor *a)
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

    Tensor *out = zf_tensor_create(a -> shape, a -> ndim);

    if (out == NULL)
    {
        return NULL;
    }

    float max_val = -FLT_MAX;
    for (int i = 0; i < a -> size; i++) 
    {
        if (a -> data[i] > max_val) max_val = a -> data[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < a -> size; i++) 
    {
        out -> data[i] = expf(a -> data[i] - max_val); // کم کردن ماکزیمم
        sum += out -> data[i];
    }

    if (sum == 0.0f)
    {
        zf_tensor_release(out);
        return NULL;
    }

    for (int i = 0; i < a -> size; i++) 
    {
        out -> data[i] /= sum;
    }

    return out;
}

