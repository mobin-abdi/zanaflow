#include <zanaflow/loss/loss.h>
#include <zanaflow/autograd/autograd.h>
#include <zanaflow/tensor/tensor.h>
#include <zanaflow/ops/ops.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

static void zf_mse_backward(AutogradNode *node, float *grad_output)
{
    if (!node || node->input_count < 2 || !grad_output)
    {
        return;
    }

    Tensor *pred = node->inputs[0];
    Tensor *target = node->inputs[1];

    if (!pred || !target || !pred->data || !target->data)
    {
        return;
    }

    if (pred->size <= 0 || pred->size != target->size)
    {
        return;
    }

    if (!zf_tensor_ensure_grad(pred))
    {
        fprintf(stderr, "Error: Could not allocate grad for pred in zf_mse_backward.\n");
        return;
    }

    float upstream = grad_output[0];
    float scale = (2.0f / (float)pred->size) * upstream;

    for (int i = 0; i < pred->size; i++)
    {
        float diff = pred->data[i] - target->data[i];
        pred->grad[i] += scale * diff;
    }
}

Tensor *zf_loss_mse(Tensor *pred, Tensor *target)
{
    int shape[1] = {1};
    Tensor *out = zf_tensor_create(shape, 1);
    if (!out)
    {
        fprintf(stderr, "zf_loss_mse: Failed to create output tensor.\n");
        return NULL;
    }

    float acc = 0.0f;
    for (int i = 0; i < pred->size; i++)
    {
        float d = pred->data[i] - target->data[i];
        acc += d * d;
    }

    out->data[0] = acc / (float)pred->size;

    out->requires_grad = pred->requires_grad;

    if (out->requires_grad)
    {
        AutogradNode *node = zf_autograd_node_create(zf_mse_backward, 2);
        if (!node)
        {
            fprintf(stderr, "zf_loss_mse: Failed to create autograd node.\n");
            zf_tensor_release(out);
            return NULL;
        }

        node->inputs[0] = pred;
        zf_tensor_retain(pred);
        node->inputs[1] = target;
        zf_tensor_retain(target);

        node->output = out;

        out->grad_node = node;
    }

    return out;
}

static void zf_cross_entropy_backward(AutogradNode *node, float *grad_output)
{
    if (!node || node->input_count < 2 || !grad_output)
        return;

    Tensor *logits = node->inputs[0];
    Tensor *target_indices = node->inputs[1];

    if (!logits || !target_indices || !logits->data || !target_indices->data)
        return;

    if (logits->ndim != 2)
    {
        fprintf(stderr, "zf_cross_entropy_backward: logits must be 2D [N,C]\n");
        return;
    }

    int N = logits->shape[0];
    int C = logits->shape[1];

    if (N <= 0 || C <= 0 || logits->size != N * C)
    {
        fprintf(stderr, "zf_cross_entropy_backward: invalid logits shape/size\n");
        return;
    }

    if (!(target_indices->size == N))
    {
        fprintf(stderr, "zf_cross_entropy_backward: target_indices size must be N\n");
        return;
    }

    if (!zf_tensor_ensure_grad(logits))
    {
        fprintf(stderr, "zf_cross_entropy_backward: could not allocate grad for logits\n");
        return;
    }

    float upstream = grad_output[0];
    float invN = 1.0f / (float)N;

    for (int n = 0; n < N; n++)
    {
        float *z = logits->data + n * C;
        float *gz = logits->grad + n * C;

        int y = (int)target_indices->data[n];
        if (y < 0 || y >= C)
        {
            fprintf(stderr, "zf_cross_entropy_backward: target index out of range at n=%d (y=%d, C=%d)\n", n, y, C);
            return;
        }

        float m = -FLT_MAX;
        for (int c = 0; c < C; c++)
        {
            if (z[c] > m)
            {
                m = z[c];
            }
        }

        float sum = 0.0f;
        for (int c = 0; c < C; c++)
        {
            sum += expf(z[c] - m);
        }

        for (int c = 0; c < C; c++)
        {
            float p = expf(z[c] - m) / sum;
            float grad = p - (c == y ? 1.0f : 0.0f);
            gz[c] += upstream * grad * invN;
        }
    }
}

Tensor *zf_loss_cross_entropy(Tensor *logits, Tensor *target_indices)
{
    if (!logits || !target_indices)
    {
        fprintf(stderr, "zf_loss_cross_entropy: logits/target_indices is NULL\n");
        return NULL;
    }

    if (!logits->data || !target_indices->data)
    {
        fprintf(stderr, "zf_loss_cross_entropy: logits/target_indices data is NULL\n");
        return NULL;
    }

    if (logits->ndim != 2)
    {
        fprintf(stderr, "zf_loss_cross_entropy: logits must be 2D [N,C]\n");
        return NULL;
    }

    int N = logits->shape[0];
    int C = logits->shape[1];

    if (N <= 0 || C <= 0 || logits->size != N * C)
    {
        fprintf(stderr, "zf_loss_cross_entropy: invalid logits shape/size\n");
        return NULL;
    }

    if (target_indices->size != N)
    {
        fprintf(stderr, "zf_loss_cross_entropy: target_indices size must equal N (got %d, expected %d)\n",
                target_indices->size, N);
        return NULL;
    }

    int shape[1] = {1};
    Tensor *out = zf_tensor_create(shape, 1);
    if (!out)
    {
        fprintf(stderr, "zf_loss_cross_entropy: Failed to create output tensor\n");
        return NULL;
    }

    float loss_acc = 0.0f;

    for (int n = 0; n < N; n++)
    {
        float *z = logits->data + n * C;
        int y = (int)target_indices->data[n];

        if (y < 0 || y >= C)
        {
            fprintf(stderr, "zf_loss_cross_entropy: target index out of range at n=%d (y=%d, C=%d)\n", n, y, C);
            zf_tensor_release(out);
            return NULL;
        }

        float m = -FLT_MAX;
        for (int c = 0; c < C; c++)
        {
            if (z[c] > m)
            {
                m = z[c];
            }
        }

        float sum = 0.0f;
        for (int c = 0; c < C; c++)
        {
            sum += expf(z[c] - m);
        }

        float logsumexp = m + logf(sum);
        float loss_n = -z[y] + logsumexp;
        loss_acc += loss_n;
    }

    out->data[0] = loss_acc / (float)N;

    out->requires_grad = logits->requires_grad;

    if (out->requires_grad)
    {
        AutogradNode *node = zf_autograd_node_create(zf_cross_entropy_backward, 2);
        if (!node)
        {
            fprintf(stderr, "zf_loss_cross_entropy: Failed to create autograd node\n");
            zf_tensor_release(out);
            return NULL;
        }

        node->inputs[0] = logits;
        zf_tensor_retain(logits);

        node->inputs[1] = target_indices;
        zf_tensor_retain(target_indices);

        node->output = out;
        out->grad_node = node;
    }

    return out;
}

static inline float zf_sigmoid_scalar(float x)
{
    if (x >= 0.0f)
    {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    else
    {
        float e = expf(x);
        return e / (1.0f + e);
    }
}

static inline float zf_bce_with_logits_elem(float z, float y)
{
   return fmaxf(z, 0.0f) - z * y + log1pf(expf(-fabsf(z)));
}

static void zf_bce_with_logits_backward(AutogradNode *node, float *grad_output)
{
    if (node == NULL || grad_output == NULL)
    {
        return;
    }

    if (node->input_count < 2)
    {
        fprintf(stderr, "zf_bce_with_logits_backward: invalid input_count\n");
        return;
    }

    Tensor *logits = node->inputs[0];
    Tensor *targets = node->inputs[1];

    if (logits == NULL || targets == NULL)
    {
        fprintf(stderr, "zf_bce_with_logits_backward: NULL inputs\n");
        return;
    }

    if (logits->data == NULL || targets->data == NULL)
    {
        fprintf(stderr, "zf_bce_with_logits_backward: NULL tensor data\n");
        return;
    }

    if (logits->size != targets->size)
    {
        fprintf(stderr, "zf_bce_with_logits_backward: size mismatch (%d vs %d)\n",
                logits->size, targets->size);
        return;
    }

    if (!zf_tensor_ensure_grad(logits))
    {
        fprintf(stderr, "zf_bce_with_logits_backward: failed to ensure logits grad\n");
        return;
    }

    float upstream = grad_output[0];
    float inv_n = 1.0f / (float)logits->size;

    for (int i = 0; i < logits->size; i++)
    {
        float z = logits->data[i];
        float y = targets->data[i];

        float p = zf_sigmoid_scalar(z);
        float dz = (p - y) * inv_n;

        logits->grad[i] += upstream * dz;
    }
}

Tensor *zf_loss_binary_cross_entropy(Tensor *logits, Tensor *targets)
{
    if (logits == NULL || targets == NULL)
    {
        fprintf(stderr, "zf_loss_bce: NULL input\n");
        return NULL;
    }

    if (logits->data == NULL || targets->data == NULL)
    {
        fprintf(stderr, "zf_loss_bce: NULL tensor data\n");
        return NULL;
    }

    if (logits->size <= 0)
    {
        fprintf(stderr, "zf_loss_bce: logits size must be > 0\n");
        return NULL;
    }

    if (logits->size != targets->size)
    {
        fprintf(stderr, "zf_loss_bce: size mismatch (%d vs %d)\n",
                logits->size, targets->size);
        return NULL;
    }

    int out_shape[1] = {1};
    Tensor *out = zf_tensor_create(out_shape, 1);
    if (out == NULL)
    {
        fprintf(stderr, "zf_loss_bce: failed to create output tensor\n");
        return NULL;
    }

    float loss_sum = 0.0f;
    for (int i = 0; i < logits->size; i++)
    {
        float z = logits->data[i];
        float y = targets->data[i];
        loss_sum += zf_bce_with_logits_elem(z, y);
    }

    out->data[0] = loss_sum / (float)logits->size;
    out->requires_grad = logits->requires_grad;

    if (out->requires_grad)
    {
        AutogradNode *node = zf_autograd_node_create(zf_bce_with_logits_backward, 2);
        if (node == NULL)
        {
            fprintf(stderr, "zf_loss_bce: failed to create autograd node\n");
            zf_tensor_release(out);
            return NULL;
        }

        node->inputs[0] = logits;
        zf_tensor_retain(logits);

        node->inputs[1] = targets;
        zf_tensor_retain(targets);

        node->output = out;
        out->grad_node = node;
    }

    return out;
}