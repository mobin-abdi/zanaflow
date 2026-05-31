#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <zanaflow/tensor/tensor.h>
#include <zanaflow/nn/init.h>

int main(void)
{
    int fan_in = 4;
    int shape_w[] = {fan_in, 3};
    int shape_b[] = {1, 3};

    Tensor *w = zf_tensor_create(shape_w, 2);
    Tensor *b = zf_tensor_create(shape_b, 2);
    assert(w && b);

    zf_init_he_uniform(w, b, fan_in);

    float limit = sqrtf(6.0f / (float)fan_in);

    for (int i = 0; i < w->size; i++)
    {
        assert(w->data[i] >= -limit && w->data[i] <= limit);
    }

    for (int i = 0; i < b->size; i++)
    {
        assert(b->data[i] == 0.0f);
    }

    zf_tensor_release(w);
    zf_tensor_release(b);

    printf("Test init_he_uniform: PASSED\n");
    return 0;
}
