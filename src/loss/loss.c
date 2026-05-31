#include <zanaflow/loss/loss.h>
#include <zanaflow/autograd/autograd.h>
#include <zanaflow/tensor/tensor.h>
#include <zanaflow/ops/ops.h>
#include <stdio.h>

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
