#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <zanaflow/tensor/tensor.h>
#include <zanaflow/loss/loss.h>
#include <zanaflow/autograd/autograd.h>

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

    pred->requires_grad = 1;
    tgt->requires_grad  = 0;

    pred->data[0] = 1.0f; tgt->data[0] = 1.0f;
    pred->data[1] = 2.0f; tgt->data[1] = 1.0f;
    pred->data[2] = 3.0f; tgt->data[2] = 2.0f;
    pred->data[3] = 4.0f; tgt->data[3] = 2.0f;

    Tensor *loss = zf_loss_mse(pred, tgt);
    assert(loss);
    assert(almost_equal(loss->data[0], 1.5f, 1e-6f));

    zf_backward(loss);

    assert(pred->grad != NULL);
    assert(almost_equal(pred->grad[0], 0.5f * (1.0f - 1.0f), 1e-5f)); // 0
    assert(almost_equal(pred->grad[1], 0.5f * (2.0f - 1.0f), 1e-5f)); // 0.5
    assert(almost_equal(pred->grad[2], 0.5f * (3.0f - 2.0f), 1e-5f)); // 0.5
    assert(almost_equal(pred->grad[3], 0.5f * (4.0f - 2.0f), 1e-5f)); // 1.0

    zf_tensor_release(loss);
    zf_tensor_release(pred);
    zf_tensor_release(tgt);

    printf("Test MSE forward+backward: PASSED\n");
    return 0;
}
