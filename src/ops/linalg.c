#include <stdlib.h>
#include <zanaflow/ops/linalg.h>

Tensor *zf_tensor_matmul_2d(const Tensor *a, const Tensor *b)
{
    if (a == NULL || b == NULL)
    {
        return NULL;
    }

    if (a -> ndim != 2 || b -> ndim != 2)
    {
        return NULL;
    }

    int m = a->shape[0];
    int n = a->shape[1];
    int n2 = b->shape[0];
    int p = b->shape[1];

    if (n != n2)
    {
        return NULL;
    }

    int out_shape[2] = { m, p };
    Tensor *out = zf_tensor_create(out_shape, 2);

    if (out == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++) 
        {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) 
            {
                float av = a->data[i * n + k];
                float bv = b->data[k * p + j];
                sum += av * bv;
            }
            out->data[i * p + j] = sum;
        }
    }

    return out;
}