#include <zanaflow/nn/init.h>
#include <zanaflow/tensor/tensor.h>
#include <stdlib.h>
#include <math.h>

static float zf_rand_uniform(float min, float max)
{
    float u = (float)rand() / (float)RAND_MAX;
    return min + (max - min) * u;
}

void zf_init_he_uniform(Tensor *weights, Tensor *bias, int fan_in)
{
    if (!weights || fan_in <= 0)
    {
        return;
    }

    float limit = sqrtf(6.0f / (float)fan_in);
    int n = zf_tensor_numel(weights);

    for (int i = 0; i < n; i++)
    {
        weights->data[i] = zf_rand_uniform(-limit, limit);
    }

    if (bias)
    {
        int bnum = zf_tensor_numel(bias);
        for (int i = 0; i < bnum; i++)
        {
            bias->data[i] = 0.0f;
        }
    }
}

void zf_init_xavier_uniform(Tensor *weights, Tensor *bias, int fan_in, int fan_out)
{
    if (!weights || fan_in <= 0 || fan_out <= 0)
    {
        return;
    }

    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    int n = zf_tensor_numel(weights);

    for (int i = 0; i < n; i++)
    {
        weights->data[i] = zf_rand_uniform(-limit, limit);
    }

    if (bias)
    {
        int bnum = zf_tensor_numel(bias);
        for (int i = 0; i < bnum; i++)
        {
            bias->data[i] = 0.0f;
        }
    }
}
