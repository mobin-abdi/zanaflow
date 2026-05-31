#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <stdbool.h>
#include <zanaflow/ops/activations.h>
#include <zanaflow/autograd/autograd.h>
#include <zanaflow/tensor/tensor.h>

Tensor *zf_relu(const Tensor *a)
{
    if (a == NULL)
    {
        return NULL;
    }

    bool requires_grad = a->requires_grad;
    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (out == NULL)
    {
        return NULL;
    }

    AutogradNode *node = NULL;
    if (requires_grad)
    {
        node = zf_autograd_node_create(zf_relu_backward, 1);
        if (!node)
        {
            zf_tensor_release(out);
            return NULL;
        }
        node->inputs[0] = (Tensor *)a;
    }
    out->requires_grad = requires_grad;
    out->grad_node = node;

    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = (a->data[i] > 0.0f) ? a->data[i] : 0.0f;
    }

    if (node)
    {
        node->output = out;
    }

    return out;
}

static void zf_relu_backward(AutogradNode *node, float *grad_output)
{
    Tensor *x = node->inputs[0];
    if (!x || !grad_output)
    {
        return;
    }

    zf_tensor_ensure_grad(x);

    for (int i = 0; i < x->size; i++)
    {
        float local = (x->data[i] > 0.0f) ? 1.0f : 0.0f;
        x->grad[i] += grad_output[i] * local;
    }
}

static void zf_sigmoid_backward(AutogradNode *node, float *grad_output)
{
    Tensor *x = node->inputs[0];
    Tensor *y = node->output;
    if (!x || !y || !grad_output)
    {
        return;
    }

    zf_tensor_ensure_grad(x);

    for (int i = 0; i < x->size; i++)
    {
        float s = y->data[i];
        x->grad[i] += grad_output[i] * (s * (1.0f - s));
    }
}

Tensor *zf_sigmoid(const Tensor *a)
{
    if (a == NULL)
    {
        return NULL;
    }

    bool requires_grad = a->requires_grad;
    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (out == NULL)
    {
        return NULL;
    }

    AutogradNode *node = NULL;
    if (requires_grad)
    {
        node = zf_autograd_node_create(zf_sigmoid_backward, 1);
        if (!node)
        {
            zf_tensor_release(out);
            return NULL;
        }
        node->inputs[0] = (Tensor *)a;

        out->grad_node = node;
    }
    out->requires_grad = requires_grad;

    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
    }

    if (node)
    {
        node->output = out;
    }

    return out;
}

Tensor *zf_leaky_relu(const Tensor *a, float alpha)
{
    if (a == NULL)
    {
        return NULL;
    }

    bool requires_grad = a->requires_grad;
    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (out == NULL)
    {
        return NULL;
    }

    AutogradNode *node = NULL;
    if (requires_grad)
    {
        node = zf_autograd_node_create(zf_leaky_relu_backward, 1);
        if (!node)
        {
            zf_tensor_release(out);
            return NULL;
        }
        node->inputs[0] = (Tensor *)a;

        float *alpha_ptr = (float *)malloc(sizeof(float));
        if (!alpha_ptr)
        {
            zf_tensor_release(out);
            zf_autograd_node_release(node);
            return NULL;
        }

        *alpha_ptr = alpha;

        node->ctx = alpha_ptr;
        node->ctx_free = free;

        out->grad_node = node;
    }
    out->requires_grad = requires_grad;

    for (int i = 0; i < a->size; i++)
    {
        float x = a->data[i];
        out->data[i] = (x > 0.0f) ? x : (alpha * x);
    }

    if (node)
    {
        node->output = out;
    }

    return out;
}

static void zf_leaky_relu_backward(AutogradNode *node, float *grad_output)
{
    Tensor *x = node->inputs[0];
    if (!x || !grad_output || !node->ctx)
    {
        return;
    }

    float alpha = *((float *)node->ctx);

    zf_tensor_ensure_grad(x);

    for (int i = 0; i < x->size; i++)
    {
        float local = (x->data[i] > 0.0f) ? 1.0f : alpha;
        x->grad[i] += grad_output[i] * local;
    }
}

static void zf_tanh_backward(AutogradNode *node, float *grad_output)
{
    Tensor *x = node->inputs[0];
    Tensor *y = node->output;
    if (!x || !y || !grad_output)
    {
        return;
    }

    zf_tensor_ensure_grad(x);

    for (int i = 0; i < x->size; i++)
    {
        float t = y->data[i];
        x->grad[i] += grad_output[i] * (1.0f - t * t);
    }
}

Tensor *zf_tanh(const Tensor *a)
{
    if (a == NULL)
    {
        return NULL;
    }

    bool requires_grad = a->requires_grad;
    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (out == NULL)
    {
        return NULL;
    }

    AutogradNode *node = NULL;
    if (requires_grad)
    {
        node = zf_autograd_node_create(zf_tanh_backward, 1);
        if (!node)
        {
            zf_tensor_release(out);
            return NULL;
        }

        node->inputs[0] = (Tensor *)a;
        out->grad_node = node;
    }
    out->requires_grad = requires_grad;

    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = tanhf(a->data[i]);
    }

    if (node)
    {
        node->output = out;
    }
    return out;
}

static void zf_softmax_backward(AutogradNode *node, float *grad_output)
{
    Tensor *x = node->inputs[0];
    Tensor *y = node->output;
    if (!x || !y || !grad_output)
    {
        return;
    }

    zf_tensor_ensure_grad(x);

    int n = x->size;
    for (int k = 0; k < n; k++)
    {
        float acc = 0.0f;
        for (int i = 0; i < n; i++)
        {
            float si = y->data[i];
            float sk = y->data[k];
            float jac = (i == k) ? (si * (1.0f - si)) : (-si * sk);
            acc += grad_output[i] * jac;
        }
        x->grad[k] += acc;
    }
}

Tensor *zf_softmax(const Tensor *a)
{
    if (a == NULL)
    {
        return NULL;
    }

    if (a->ndim != 1)
    {
        fprintf(stderr, "Error: Softmax input must be 1D.\n");
        return NULL;
    }

    if (a->size <= 0)
    {
        fprintf(stderr, "Error: Softmax input cannot be empty.\n");
        return NULL;
    }

    bool requires_grad = a->requires_grad;
    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (out == NULL)
    {
        return NULL;
    }

    AutogradNode *node = NULL;
    if (requires_grad)
    {
        node = zf_autograd_node_create(zf_softmax_backward, 1);
        if (!node)
        {
            zf_tensor_release(out);
            return NULL;
        }
        node->inputs[0] = (Tensor *)a;

        out->grad_node = node;
    }
    out->requires_grad = requires_grad;

    float max_val = -FLT_MAX;
    for (int i = 0; i < a->size; i++)
    {
        if (a->data[i] > max_val)
        {
            max_val = a->data[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < a->size; i++)
    {
        out->data[i] = expf(a->data[i] - max_val);
        sum += out->data[i];
    }

    if (sum == 0.0f)
    {
        fprintf(stderr, "Error: Softmax sum is zero, cannot normalize.\n");
        zf_tensor_release(out);
        if (node)
        {
            zf_autograd_node_release(node);
        }
        return NULL;
    }

    for (int i = 0; i < a->size; i++)
    {
        out->data[i] /= sum;
    }

    if (node)
    {
        node->output = out;
    }

    return out;
}