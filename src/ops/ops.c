#include <zanaflow/ops/ops.h>
#include <zanaflow/autograd/autograd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int zf_tensor_same_shape(const Tensor *a, const Tensor *b) 
{
    if (!a || !b)
    {
        return 0;
    }

    if (a->ndim != b->ndim)
    {
        return 0;
    }

    for (int i = 0; i < a->ndim; i++) 
    {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    return 1;
}

static void backward_add(AutogradNode *node, float *grad_out) 
{
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    if (!grad_out) { fprintf(stderr, "backward_add: grad_out is NULL\n"); return; }
    if (a->requires_grad) 
    {
        zf_tensor_ensure_grad(a);
        if (!a->grad)
        {
            fprintf(stderr, "backward_add: a->grad is NULL after ensure_grad\n");
            return;
        }
        for (int i = 0; i < a->size; i++) a->grad[i] += grad_out[i]; // درسته
    }
    if (b->requires_grad) 
    {
        zf_tensor_ensure_grad(b);
        if (!a->grad)
        {
            fprintf(stderr, "backward_add: a->grad is NULL after ensure_grad\n");
            return;
        }
        for (int i = 0; i < b->size; i++) b->grad[i] += grad_out[i]; // درسته
    }
}


static void backward_sub(AutogradNode *node, float *grad_out) 
{
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    if (!grad_out) 
    {
        fprintf(stderr, "backward_sub: grad_out is NULL\n"); return; 
    }

    if (a->requires_grad) 
    {
        zf_tensor_ensure_grad(a);
                if (!a->grad)
        {
            fprintf(stderr, "backward_add: a->grad is NULL after ensure_grad\n");
            return;
        }
        for (int i = 0; i < a->size; i++) a->grad[i] += grad_out[i];
    }

    if (b->requires_grad) 
    {
        zf_tensor_ensure_grad(b);
                if (!a->grad)
        {
            fprintf(stderr, "backward_add: a->grad is NULL after ensure_grad\n");
            return;
        }
        for (int i = 0; i < b->size; i++) b->grad[i] -= grad_out[i];
    }
}

static void backward_mul(AutogradNode *node, float *grad_out) 
{
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    if (!grad_out)
    {
        fprintf(stderr, "backward_mul: grad_out is NULL\n"); return; 
    }

    if (a->requires_grad) 
    {
        zf_tensor_ensure_grad(a);
        if (!a->grad)
        {
            fprintf(stderr, "backward_add: a->grad is NULL after ensure_grad\n");
            return;
        }
        for (int i = 0; i < a->size; i++) 
        {
            a->grad[i] += grad_out[i] * b->data[i]; // این هم درسته!
        }
    }
    if (b->requires_grad) 
    {
        zf_tensor_ensure_grad(b);
        if (!a->grad)
        {
            fprintf(stderr, "backward_add: a->grad is NULL after ensure_grad\n");
            return;
        }
        for (int i = 0; i < b->size; i++) 
        {
            b->grad[i] += grad_out[i] * a->data[i]; // این هم درسته!
        }
    }
}

static void backward_sum_all(AutogradNode *node, float *grad_out) 
{
    Tensor *a = node->inputs[0];
    if (!grad_out)
    {
        fprintf(stderr, "backward_sum_all: grad_out is NULL\n"); return;
    }
    
    if (a->requires_grad) 
    {
        zf_tensor_ensure_grad(a);
        if (!a->grad)
        {
            fprintf(stderr, "backward_add: a->grad is NULL after ensure_grad\n");
            return;
        }
        float go = grad_out[0];
        for (int i = 0; i < a->size; i++) 
        {
            a->grad[i] += go;
        }
    }
}

static void backward_mean_all(AutogradNode *node, float *grad_out) 
{
    Tensor *a = node->inputs[0];
    if (!grad_out) 
    {
        fprintf(stderr, "backward_mean_all: grad_out is NULL\n"); return;
    }
    
    if (a->requires_grad) 
    {
        zf_tensor_ensure_grad(a);
        if (!a->grad)
        {
            fprintf(stderr, "backward_add: a->grad is NULL after ensure_grad\n");
            return;
        }
        float go = grad_out[0];
        float inv_n = 1.0f / (float)a->size;
        for (int i = 0; i < a->size; i++) 
        {
            a->grad[i] += go * inv_n;
        }
    }
}

static void backward_add_bias(AutogradNode *node, float *grad_out)
{
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    if (!grad_out) 
    {
        fprintf(stderr, "backward_add_bias: grad_out is NULL\n"); return;
    }

    int rows = a->shape[0];
    int cols = a->shape[1];

    if (a->requires_grad) 
    {
        zf_tensor_ensure_grad(a);
        if (!a->grad)
        {
            fprintf(stderr, "backward_add: a->grad is NULL after ensure_grad\n");
            return;
        }

        for (int r = 0; r < rows; r++) 
        {
            for (int c = 0; c < cols; c++) 
            {
                a->grad[r * cols + c] += grad_out[r * cols + c];
            }
        }
    }

    if (b->requires_grad) 
    {
        zf_tensor_ensure_grad(b);
        if (!a->grad)
        {
            fprintf(stderr, "backward_add: a->grad is NULL after ensure_grad\n");
            return;
        }
        for (int j = 0; j < cols; j++) 
        {
            float acc = 0.0f;
            for (int r = 0; r < rows; r++) 
            {
                acc += grad_out[r * cols + j];
            }
            b->grad[j] += acc;
        }
    }
}

Tensor *zf_tensor_add(Tensor *a, Tensor *b)
{
    if (!a || !b)
    {
        return NULL;
    }

    if (!zf_tensor_same_shape(a, b))
    {
        return NULL;
    }

    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (!out)
    {
        return NULL;
    }

    for (int i = 0; i < out->size; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }

    if (a->requires_grad || b->requires_grad)
    {
        out->requires_grad = 1;
        out->grad_node = zf_autograd_node_create(backward_add, 2);
        if (!out->grad_node) 
        {
            zf_tensor_release(out);
            return NULL;
        }

        out->grad_node->inputs[0] = a; zf_tensor_retain(a);
        out->grad_node->inputs[1] = b; zf_tensor_retain(b);

        out->grad_node->ctx = out;
        out->grad_node->ctx_free = NULL;
    }

    return out;
}

Tensor *zf_tensor_sub(Tensor *a, Tensor *b) 
{
    if (!a || !b)
    {
        return NULL;
    }

    if (!zf_tensor_same_shape(a, b))
    {
        return NULL;
    }

    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (!out)
    {
        return NULL;
    }

    for (int i = 0; i < out->size; i++)
    {
        out->data[i] = a->data[i] - b->data[i];

    }

    if (a->requires_grad || b->requires_grad) 
    {
        out->requires_grad = 1;
        out->grad_node = zf_autograd_node_create(backward_sub, 2);

        if (!out->grad_node)
        {
            zf_tensor_release(out);
            return NULL;
        }

        out->grad_node->inputs[0] = a; zf_tensor_retain(a);
        out->grad_node->inputs[1] = b; zf_tensor_retain(b);

        out->grad_node->ctx = out;
        out->grad_node->ctx_free = NULL;
    }

    return out;
}

Tensor *zf_tensor_mul(Tensor *a, Tensor *b)
{
    if (!a || !b)
    {
        return NULL;
    }
    if (!zf_tensor_same_shape(a, b))
    {
        return NULL;
    }

    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (!out)
    {
        return NULL;
    }

    for (int i = 0; i < out->size; i++)
    {
        out->data[i] = a->data[i] * b->data[i];
    }

    if (a->requires_grad || b->requires_grad) 
    {
        out->requires_grad = 1;
        out->grad_node = zf_autograd_node_create(backward_mul, 2);
        if (!out->grad_node) {
            zf_tensor_release(out);
            return NULL;
        }

        out->grad_node->inputs[0] = a; zf_tensor_retain(a);
        out->grad_node->inputs[1] = b; zf_tensor_retain(b);

        out->grad_node->ctx = out;
        out->grad_node->ctx_free = NULL;
    }

    return out;
}

Tensor *zf_tensor_sum_all(Tensor *a) 
{
    if (!a)
    {
        return NULL;
    }

    int s[1] = {1};
    Tensor *out = zf_tensor_create(s, 1);
    if (!out)
    {
        return NULL;
    }

    float acc = 0.0f;
    for (int i = 0; i < a->size; i++) acc += a->data[i];
    out->data[0] = acc;

    if (a->requires_grad) 
    {
        out->requires_grad = 1;
        out->grad_node = zf_autograd_node_create(backward_sum_all, 1);
        if (!out->grad_node)
        {
            zf_tensor_release(out);
            return NULL;
        }

        out->grad_node->inputs[0] = a; zf_tensor_retain(a);

        out->grad_node->ctx = out;
        out->grad_node->ctx_free = NULL;
    }

    return out;
}

Tensor *zf_tensor_mean_all(Tensor *a) 
{
    if (!a)
    {
        return NULL;
    }
    if (a->size <= 0)
    {
        return NULL;
    }

    int s[1] = {1};
    Tensor *out = zf_tensor_create(s, 1);
    if (!out)
    {
        return NULL;
    }

    float acc = 0.0f;
    for (int i = 0; i < a->size; i++) acc += a->data[i];
    out->data[0] = acc / (float)a->size;

    if (a->requires_grad) 
    {
        out->requires_grad = 1;
        out->grad_node = zf_autograd_node_create(backward_mean_all, 1);
        if (!out->grad_node)
        {
            zf_tensor_release(out);
            return NULL;
        }

        out->grad_node->inputs[0] = a; zf_tensor_retain(a);

        out->grad_node->ctx = out;
        out->grad_node->ctx_free = NULL;
    }

    return out;
}

Tensor *zf_tensor_add_bias(Tensor *a, Tensor *b) 
{
    if (!a || !b)
    {
        return NULL;
    }
    if (a->ndim != 2)
    {
        return NULL;
    }
    if (b->ndim != 1)
    {
        return NULL;
    }

    int rows = a->shape[0];
    int cols = a->shape[1];
    if (b->shape[0] != cols)
    {
        return NULL;
    }

    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (!out)
    {
        return NULL;
    }

    for (int r = 0; r < rows; r++) 
    {
        for (int c = 0; c < cols; c++) 
        {
            int idx = r * cols + c;
            out->data[idx] = a->data[idx] + b->data[c];
        }
    }

    if (a->requires_grad || b->requires_grad) 
    {
        out->requires_grad = 1;
        out->grad_node = zf_autograd_node_create(backward_add_bias, 2);
        if (!out->grad_node) {
            zf_tensor_release(out);
            return NULL;
        }

        out->grad_node->inputs[0] = a; zf_tensor_retain(a);
        out->grad_node->inputs[1] = b; zf_tensor_retain(b);

        out->grad_node->ctx = out;
        out->grad_node->ctx_free = NULL;
    }

    return out;
}

void zf_tensor_add_inplace(Tensor *a, const Tensor *b) 
{
    if (!a || !b)
    {
        return;
    }
    if (!zf_tensor_same_shape(a, b))
    {
        return;
    }

    for (int i = 0; i < a->size; i++)
    {
        a->data[i] += b->data[i];
    }
}
Tensor *zf_tensor_sum_rows(const Tensor *a) {
    if (!a)
    {
        return NULL;
    }

    if (a->ndim != 2)
    {
        return NULL;
    }

    int rows = a->shape[0];
    int cols = a->shape[1];

    int out_shape[1] = { rows };
    Tensor *out = zf_tensor_create(out_shape, 1);
    
    if (!out)
    {
        return NULL;
    }

    for (int r = 0; r < rows; r++) 
    {
        float acc = 0.0f;
        for (int c = 0; c < cols; c++)
        {
            acc += a->data[r * cols + c];
        }
        out->data[r] = acc;
    }
    return out;
}

Tensor *zf_tensor_mat_mul(const Tensor *a, const Tensor *b) {
    if (!a || !b)
    {
        return NULL;
    }

    if (a->ndim != 2 || b->ndim != 2)
    {
        return NULL;
    }

    int m = a->shape[0];
    int k = a->shape[1];
    int k2 = b->shape[0];
    int n = b->shape[1];

    if (k != k2)
    {
        return NULL;
    }

    int out_shape[2] = { m, n };
    Tensor *out = zf_tensor_create(out_shape, 2);

    if (!out)
    {
        return NULL;
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float acc = 0.0f;
            for (int t = 0; t < k; t++)
            {
                acc += a->data[i * k + t] * b->data[t * n + j];
            }
            out->data[i * n + j] = acc;
        }
    }
    return out;
}

Tensor *zf_tensor_mat_mul_transpose(const Tensor *a, const Tensor *b) {
    if (!a || !b)
    {
        return NULL;
    }

    if (a->ndim != 2 || b->ndim != 2)
    {
        return NULL;
    }

    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[0];
    int k2 = b->shape[1];

    if (k != k2)
    {
        return NULL;
    }

    int out_shape[2] = { m, n };
    Tensor *out = zf_tensor_create(out_shape, 2);
    if (!out)
    {
        return NULL;
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float acc = 0.0f;
            for (int t = 0; t < k; t++)
            {
                acc += a->data[i * k + t] * b->data[j * k + t];
            }
            out->data[i * n + j] = acc;
        }
    }
    return out;
}
