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
        if (a->shape[i] != b->shape[i])
            return 0;
    }
    return 1;
}

static void backward_add(AutogradNode *node, float *grad_out)
{
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];

    if (!grad_out)
    {
        fprintf(stderr, "backward_add: grad_out is NULL\n");
        return;
    }

    if (a && a->requires_grad)
    {
        zf_tensor_ensure_grad(a);
        if (!a->grad)
        {
            fprintf(stderr, "backward_add: a->grad is NULL after ensure_grad\n");
            return;
        }
        for (int i = 0; i < a->size; i++)
        {
            a->grad[i] += grad_out[i];
        }
    }

    if (b && b->requires_grad)
    {
        zf_tensor_ensure_grad(b);
        if (!b->grad)
        {
            fprintf(stderr, "backward_add: b->grad is NULL after ensure_grad\n");
            return;
        }
        for (int i = 0; i < b->size; i++)
        {
            b->grad[i] += grad_out[i];
        }
    }
}

static void backward_sub(AutogradNode *node, float *grad_out)
{
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    if (!grad_out)
    {
        fprintf(stderr, "backward_sub: grad_out is NULL\n");
        return;
    }

    if (a && a->requires_grad)
    {
        zf_tensor_ensure_grad(a);
        if (!a->grad)
        {
            fprintf(stderr, "backward_sub: a->grad is NULL after ensure_grad\n");
            return;
        }
        for (int i = 0; i < a->size; i++)
            a->grad[i] += grad_out[i];
    }

    if (b && b->requires_grad)
    {
        zf_tensor_ensure_grad(b);
        if (!b->grad)
        {
            fprintf(stderr, "backward_sub: b->grad is NULL after ensure_grad\n"); // Corrected error message
            return;
        }
        for (int i = 0; i < b->size; i++)
            b->grad[i] -= grad_out[i];
    }
}

static void backward_mat_mul(AutogradNode *node, float *grad_out)
{
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];

    if (!grad_out)
    {
        fprintf(stderr, "backward_mat_mul: grad_out is NULL\n");
        return;
    }

    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];

    if (a && a->requires_grad)
    {
        zf_tensor_ensure_grad(a);
        if (!a->grad)
        {
            fprintf(stderr, "backward_mat_mul: a->grad is NULL after ensure_grad\n");
            return;
        }
        for (int i = 0; i < m; i++)
        {
            for (int t = 0; t < k; t++)
            {
                float acc = 0.0f;
                for (int j = 0; j < n; j++)
                {
                    acc += grad_out[i * n + j] * b->data[t * n + j];
                }
                a->grad[i * k + t] += acc;
            }
        }
    }

    if (b && b->requires_grad)
    {
        zf_tensor_ensure_grad(b);
        if (!b->grad)
        {
            fprintf(stderr, "backward_mat_mul: b->grad is NULL after ensure_grad\n");
            return;
        }
        for (int t = 0; t < k; t++)
        {
            for (int j = 0; j < n; j++)
            {
                float acc = 0.0f;
                for (int i = 0; i < m; i++)
                {
                    acc += a->data[i * k + t] * grad_out[i * n + j];
                }
                b->grad[t * n + j] += acc;
            }
        }
    }
}

static void backward_mul(AutogradNode *node, float *grad_out)
{
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];

    if (!grad_out)
    {
        fprintf(stderr, "backward_mul: grad_out is NULL\n");
        return;
    }

    if (a && a->requires_grad)
    {
        zf_tensor_ensure_grad(a);
        if (!a->grad)
        {
            fprintf(stderr, "backward_mul: a->grad is NULL after ensure_grad\n");
            return;
        }
        for (int i = 0; i < a->size; i++)
        {
            a->grad[i] += grad_out[i] * b->data[i];
        }
    }

    if (b && b->requires_grad)
    {
        zf_tensor_ensure_grad(b);
        if (!b->grad)
        {
            fprintf(stderr, "backward_mul: b->grad is NULL after ensure_grad\n");
            return;
        }
        for (int i = 0; i < b->size; i++)
        {
            b->grad[i] += grad_out[i] * a->data[i];
        }
    }
}

static void backward_sum_all(AutogradNode *node, float *grad_out)
{
    Tensor *a = node->inputs[0];
    if (!grad_out)
    {
        fprintf(stderr, "backward_sum_all: grad_out is NULL\n");
        return;
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
        fprintf(stderr, "backward_mean_all: grad_out is NULL\n");
        return;
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
        fprintf(stderr, "backward_add_bias: grad_out is NULL\n");
        return;
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
        if (!b->grad)
        {
            fprintf(stderr, "backward_add: b->grad is NULL after ensure_grad\n");
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
        fprintf(stderr, "zf_tensor_add: NULL input tensor.\n");
        return NULL;
    }
    if (!zf_tensor_same_shape(a, b))
    {
        fprintf(stderr, "zf_tensor_add: shape mismatch (%d dims, %d dims).\n", a->ndim, b->ndim);
        return NULL;
    }

    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (!out)
    {
        fprintf(stderr, "zf_tensor_add: Failed to create output tensor.\n");
        return NULL;
    }

    for (int i = 0; i < out->size; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }

    out->requires_grad = (a->requires_grad || b->requires_grad);

    if (out->requires_grad)
    {
        AutogradNode *node = zf_autograd_node_create(backward_add, 2);
        if (!node)
        {
            fprintf(stderr, "zf_tensor_add: Failed to create autograd node.\n");
            zf_tensor_release(out);
            return NULL;
        }

        node->inputs[0] = a;
        zf_tensor_retain(a);
        node->inputs[1] = b;
        zf_tensor_retain(b);

        node->output = out;
        node->ctx = NULL;
        node->ctx_free = NULL;

        out->grad_node = node;
    }

    return out;
}

Tensor *zf_tensor_sub(Tensor *a, Tensor *b)
{
    if (!a || !b)
    {
        fprintf(stderr, "zf_tensor_sub: NULL input tensor.\n");
        return NULL;
    }
    if (!zf_tensor_same_shape(a, b))
    {
        fprintf(stderr, "zf_tensor_sub: shape mismatch.\n");
        return NULL;
    }

    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (!out)
    {
        fprintf(stderr, "zf_tensor_sub: Failed to create output tensor.\n");
        return NULL;
    }

    for (int i = 0; i < out->size; i++)
    {
        out->data[i] = a->data[i] - b->data[i];
    }

    out->requires_grad = (a->requires_grad || b->requires_grad);

    if (out->requires_grad)
    {
        AutogradNode *node = zf_autograd_node_create(backward_sub, 2);
        if (!node)
        {
            fprintf(stderr, "zf_tensor_sub: Failed to create autograd node.\n");
            zf_tensor_release(out);
            return NULL;
        }

        node->inputs[0] = a;
        zf_tensor_retain(a);
        node->inputs[1] = b;
        zf_tensor_retain(b);

        node->output = out;
        node->ctx = NULL;
        node->ctx_free = NULL;

        out->grad_node = node;
    }

    return out;
}

Tensor *zf_tensor_mul(Tensor *a, Tensor *b)
{
    if (!a || !b)
    {
        fprintf(stderr, "zf_tensor_mul: NULL input tensor.\n");
        return NULL;
    }
    if (!zf_tensor_same_shape(a, b))
    {
        fprintf(stderr, "zf_tensor_mul: shape mismatch.\n");
        return NULL;
    }

    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (!out)
    {
        fprintf(stderr, "zf_tensor_mul: Failed to create output tensor.\n");
        return NULL;
    }

    for (int i = 0; i < out->size; i++)
    {
        out->data[i] = a->data[i] * b->data[i];
    }

    out->requires_grad = (a->requires_grad || b->requires_grad);

    if (out->requires_grad)
    {
        AutogradNode *node = zf_autograd_node_create(backward_mul, 2);
        if (!node)
        {
            fprintf(stderr, "zf_tensor_mul: Failed to create autograd node.\n");
            zf_tensor_release(out);
            return NULL;
        }

        node->inputs[0] = a;
        zf_tensor_retain(a);
        node->inputs[1] = b;
        zf_tensor_retain(b);

        node->output = out;
        node->ctx = NULL;
        node->ctx_free = NULL;

        out->grad_node = node;
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
    for (int i = 0; i < a->size; i++)
    {
        acc += a->data[i];
    }
    out->data[0] = acc;

    if (a->requires_grad)
    {
        out->requires_grad = 1;

        AutogradNode *node = zf_autograd_node_create(backward_sum_all, 1);
        if (!node)
        {
            zf_tensor_release(out);
            return NULL;
        }

        node->inputs[0] = a;
        zf_tensor_retain(a);

        node->output = out;
        node->ctx = NULL;
        node->ctx_free = NULL;

        out->grad_node = node;
    }

    return out;
}

Tensor *zf_tensor_mean_all(Tensor *a)
{
    if (!a)
        return NULL;
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
    for (int i = 0; i < a->size; i++)
    {
        acc += a->data[i];
    }

    out->data[0] = acc / (float)a->size;

    if (a->requires_grad)
    {
        out->requires_grad = 1;

        AutogradNode *node = zf_autograd_node_create(backward_mean_all, 1);
        if (!node)
        {
            zf_tensor_release(out);
            return NULL;
        }

        node->inputs[0] = a;
        zf_tensor_retain(a);

        node->output = out;
        node->ctx = NULL;  
        node->ctx_free = NULL;

        out->grad_node = node;
    }

    return out;
}

Tensor *zf_tensor_add_bias(Tensor *a, Tensor *b)
{
    if (!a || !b) return NULL;
    if (a->ndim != 2) return NULL;

    int rows = a->shape[0];
    int cols = a->shape[1];
    int bias_size;

    // پشتیبانی از بایاس ۱ بعدی و ۲ بعدی [1, cols]
    if (b->ndim == 1) {
        if (b->shape[0] != cols) return NULL;
        bias_size = b->shape[0];
    } else if (b->ndim == 2) {
        if (b->shape[0] != 1 || b->shape[1] != cols) return NULL;
        bias_size = b->shape[1];  // = cols
    } else {
        return NULL;
    }

    Tensor *out = zf_tensor_create(a->shape, a->ndim);
    if (!out) return NULL;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            out->data[idx] = a->data[idx] + b->data[c];  // b->data پیوسته است
        }
    }

    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = 1;
        AutogradNode *node = zf_autograd_node_create(backward_add_bias, 2);
        if (!node) {
            zf_tensor_release(out);
            return NULL;
        }
        node->inputs[0] = a;
        zf_tensor_retain(a);
        node->inputs[1] = b;
        zf_tensor_retain(b);
        node->output = out;
        node->ctx = NULL;
        node->ctx_free = NULL;
        out->grad_node = node;
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
Tensor *zf_tensor_sum_rows(const Tensor *a)
{
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

    int out_shape[1] = {rows};
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

Tensor *zf_tensor_mat_mul(Tensor *a, Tensor *b)
{
    if (!a || !b)
    {
        fprintf(stderr, "zf_tensor_mat_mul: NULL input tensor.\n");
        return NULL;
    }
    if (a->ndim != 2 || b->ndim != 2)
    {
        fprintf(stderr, "zf_tensor_mat_mul: Input tensors must be 2D.\n");
        return NULL;
    }

    int m = a->shape[0];
    int k = a->shape[1];
    int k2 = b->shape[0];
    int n = b->shape[1];

    if (k != k2)
    {
        fprintf(stderr, "zf_tensor_mat_mul: Shape mismatch for multiplication (%d != %d).\n", k, k2);
        return NULL;
    }

    int out_shape[2] = {m, n};
    Tensor *out = zf_tensor_create(out_shape, 2);
    if (!out)
    {
        fprintf(stderr, "zf_tensor_mat_mul: Failed to create output tensor.\n");
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

    out->requires_grad = (a->requires_grad || b->requires_grad);
    if (out->requires_grad)
    {
        AutogradNode *node = zf_autograd_node_create(backward_mat_mul, 2);
        if (!node)
        {
            fprintf(stderr, "zf_tensor_mat_mul: Failed to create autograd node.\n");
            zf_tensor_release(out);
            return NULL;
        }

        node->inputs[0] = a;
        zf_tensor_retain(a);
        node->inputs[1] = b;
        zf_tensor_retain(b);

        node->output = out;
        node->ctx = NULL;
        node->ctx_free = NULL;

        out->grad_node = node;
    }

    return out;
}

Tensor *zf_tensor_mat_mul_transpose(const Tensor *a, const Tensor *b)
{
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

    int out_shape[2] = {m, n};
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
