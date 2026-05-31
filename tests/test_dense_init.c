#include <zanaflow/nn/dense.h>
#include <zanaflow/tensor/tensor.h>

#include <stdio.h>
#include <stdlib.h>

static void print_tensor_1d(const char *name, const Tensor *t)
{
    printf("%s shape=[%d]\n", name, t->shape[0]);
    for (int i = 0; i < t->shape[0]; i++)
    {
        printf("  %s[%d] = %.6f\n", name, i, t->data[i]);
    }
}

static void print_tensor_2d(const char *name, const Tensor *t)
{
    int r = t->shape[0];
    int c = t->shape[1];

    printf("%s shape=[%d,%d]\n", name, r, c);
    for (int i = 0; i < r; i++)
    {
        printf("  %s[%d,:] ", name, i);
        for (int j = 0; j < c; j++)
        {
            int idx = i * c + j;
            printf("% .6f", t->data[idx]);
            if (j + 1 < c) printf(", ");
        }
        printf("\n");
    }
}

int main(void)
{
    srand(42);

    int in_features = 3;
    int out_features = 2;

    DenseLayer *layer = zf_dense_create(in_features, out_features);
    if (!layer)
    {
        fprintf(stderr, "Failed to create DenseLayer\n");
        return 1;
    }

    print_tensor_2d("weights", layer->weights);
    print_tensor_1d("bias", layer->bias);

    for (int i = 0; i < layer->bias->shape[0]; i++)
    {
        if (layer->bias->data[i] != 0.0f)
        {
            fprintf(stderr, "Bias is not zero-initialized at %d: %f\n", i, layer->bias->data[i]);
            zf_dense_free(layer);
            return 2;
        }
    }

    zf_dense_free(layer);
    return 0;
}
