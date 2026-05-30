#include <zanaflow/tensor/tensor.h>
#include <zanaflow/autograd/autograd.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tensor *zf_tensor_create(int *shape, int ndim)
{
    Tensor *t = malloc(sizeof(Tensor));
    if (!t)
    {
        return NULL;
    }

    t->ndim = ndim;
    t->size = 1;
    t->shape = malloc(ndim * sizeof(int));
    if (!t->shape)
    {
        free(t);
        return NULL;
    } 

    for (int i = 0; i < ndim; i++) 
    {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    t->data = calloc(t->size, sizeof(float));
    if (!t->data)
    {
        free(t->shape); 
        free(t); return NULL;
    } 

    t->grad = NULL;
    t->grad_node = NULL;
    t->requires_grad = 0;
    ref_init(&t->ref);
    return t;
}

void zf_tensor_retain(Tensor *t) 
{
    if (t)
    {
        ref_retain(&t->ref);
    }
}

void zf_tensor_release(Tensor *t) 
{
    if (!t)
    {
        return;
    }
    if (ref_release(&t->ref))
    {
        free(t->data);
        free(t->shape);
        if (t->grad)
        {
            free(t->grad);
        }
        if (t->grad_node)
        {
            zf_autograd_node_release(t->grad_node);
        }
        free(t);
    }
}

int zf_tensor_ensure_grad(Tensor *t) 
{
    if (!t)
    {
        return 0;
    }
    if (t->grad)
    {
        return 1;
    }
    t->grad = calloc(t->size, sizeof(float));
    return t->grad != NULL;
}

void zf_tensor_zero_grad(Tensor *t)
{
    if (t && t->grad) 
    {
        memset(t->grad, 0, t->size * sizeof(float));
    }
}

void zf_tensor_print(Tensor *t) 
{
    if (!t)
    {
        printf("NULL Tensor\n"); return;
    }
    printf("Tensor(shape=[");
    for(int i = 0; i < t->ndim; ++i) 
    {
        printf("%d%s", t->shape[i], (i == t->ndim - 1) ? "" : ", ");
    }
    printf("], size=%d, requires_grad=%d): [", t->size, t->requires_grad);
    for (int i = 0; i < t->size; i++) 
    {
        printf("%.4f%s", t->data[i], (i == t->size - 1) ? "" : ", ");
    }
    printf("]\n");
    if (t->grad) 
    {
        printf("  Grad: [");
        for (int i = 0; i < t->size; i++) 
        {
            printf("%.4f%s", t->grad[i], (i == t->size - 1) ? "" : ", ");
        }
        printf("]\n");
    }
}

Tensor *zf_tensor_clone(const Tensor *t) 
{
    if (!t)
    {
        return NULL;
    }

    Tensor *clone = zf_tensor_create(t->shape, t->ndim);
    if (!clone)
    {
        return NULL;
    }

    memcpy(clone->data, t->data, t->size * sizeof(float));

    clone->requires_grad = t->requires_grad;

    return clone;
}

void zf_tensor_fill(Tensor *t, float value)
{
    if (!t || !t->data) 
    {
        return;
    }
    for (int i = 0; i < t->size; ++i)
    {
        t->data[i] = value;
    }
}

Tensor *zf_tensor_full(const int *shape, int ndim, float value)
{
    Tensor *t = zf_tensor_create((int*)shape, ndim);
    
    if (!t)
    {
        return NULL;
    }

    if (value == 0.0f)
    {
        memset(t->data, 0, sizeof(float) * t->size);
    }
    else
    {
        zf_tensor_fill(t, value);
    }
}

Tensor *zf_tensor_zeros(const int *shape, int ndim)
{
    return zf_tensor_full(shape, ndim, 0.0f);
}

Tensor *zf_tensor_ones(const int *shape, int ndim)
{
    return zf_tensor_full(shape, ndim, 1.0f);
}

Tensor *zf_tensor_reshape(const Tensor *t, const int *new_shape, int new_ndim)
{
    if (!t)
    {
        return NULL;
    }

    int new_size = 1;
    
    for (int i = 0; i < new_ndim; i++)
    {
        new_size *= new_shape[i];
    }
    
    if (new_size != t->size) 
    {
        fprintf(stderr, "Error: Reshape size mismatch.\n");
        return NULL;
    }

    Tensor *out = zf_tensor_create((int*)new_shape, new_ndim);
    if (!out)
    {
        return NULL;
    }    
    
    memcpy(out->data, t->data, sizeof(float) * t->size);
    
    return out;
}
