#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <zanaflow/tensor/tensor.h>
#include <zanaflow/loss/loss.h>

static int almost_equal(float a, float b, float eps)
{
    return fabsf(a - b) < eps;
}

int main(void)
{
    int shape[] = {4};

    Tensor *pred = zf_tensor_create(shape, 1);
    Tensor *tgt  = zf_tensor_create(shape, 1);
    assert(pred && tgt);

    pred->data[0] = 1.0f; tgt->data[0] = 1.0f;
    pred->data[1] = 2.0f; tgt->data[1] = 1.0f;
    pred->data[2] = 3.0f; tgt->data[2] = 2.0f;
    pred->data[3] = 4.0f; tgt->data[3] = 2.0f;

    float loss = zf_loss_mse(pred, tgt);
    assert(almost_equal(loss, 1.5f, 1e-6f));

    zf_tensor_release(pred);
    zf_tensor_release(tgt);

    printf("Test MSE: PASSED\n");
    return 0;
}
