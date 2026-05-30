#include <stdlib.h>
#include <zanaflow/loss/loss.h>
#include <zanaflow/ops/ops.h>

float zf_loss_mse(const Tensor *pred, const Tensor *target)
{
    if (pred == NULL || target == NULL)
    {
        return -1.0f;
    }

    if (!zf_tensor_same_shape(pred, target))
    {
        return -1.0f;
    }

    if (pred -> size <= 0)
    {
        return -1.0f;
    }

    float accuracy = 0.0f;

    for (int i = 0; i < pred -> size; i++)
    {
        float d = pred -> data[i] - target -> data[i];
        accuracy += d * d;
    }

    return accuracy / (float)pred -> size;
}

