#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <zanaflow/tensor/tensor.h>
#include <zanaflow/autograd/autograd.h>
#include <zanaflow/loss/loss.h>

static void zero_grad(Tensor *t)
{
    if (t && t->grad)
        memset(t->grad, 0, sizeof(float) * (size_t)t->size);
}

static float sigmoidf_stable(float x)
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

static void linear_forward_or(const float *X, int N, const Tensor *W, const Tensor *b, Tensor *out_logits)
{
    for (int n = 0; n < N; n++)
    {
        float x0 = X[n * 2 + 0];
        float x1 = X[n * 2 + 1];
        out_logits->data[n] = x0 * W->data[0] + x1 * W->data[1] + b->data[0];
    }
}

int main(void)
{
    float X[8] = {
        0, 0,
        0, 1,
        1, 0,
        1, 1};
    float y[4] = {0, 1, 1, 1};
    int N = 4;

    int wshape[1] = {2};
    Tensor *W = zf_tensor_create(wshape, 1);
    assert(W && W->data);
    W->requires_grad = 1;

    int bshape[1] = {1};
    Tensor *b = zf_tensor_create(bshape, 1);
    assert(b && b->data);
    b->requires_grad = 1;

    int lshape[1] = {N};
    Tensor *logits = zf_tensor_create(lshape, 1);
    assert(logits && logits->data);
    logits->requires_grad = 1; // MUST be set before building loss

    Tensor *targets = zf_tensor_create(lshape, 1);
    assert(targets && targets->data);
    targets->requires_grad = 0;
    memcpy(targets->data, y, sizeof(y));

    W->data[0] = 0.0f;
    W->data[1] = 0.0f;
    b->data[0] = 0.0f;

    float lr = 0.5f;
    int steps = 2000;

    for (int step = 0; step < steps; step++)
    {
        linear_forward_or(X, N, W, b, logits);

        zero_grad(logits);
        zero_grad(W);
        zero_grad(b);

        Tensor *loss = zf_loss_binary_cross_entropy(logits, targets);
        assert(loss && loss->data);

        zf_backward(loss);

        float dW0 = 0.0f, dW1 = 0.0f, dbv = 0.0f;
        for (int n = 0; n < N; n++)
        {
            float dz = logits->grad[n];
            dW0 += X[n * 2 + 0] * dz;
            dW1 += X[n * 2 + 1] * dz;
            dbv += dz;
        }

        W->data[0] -= lr * dW0;
        W->data[1] -= lr * dW1;
        b->data[0] -= lr * dbv;

        if (step % 200 == 0 || step == steps - 1)
        {
            printf("step=%d loss=%.6f W=[%.4f %.4f] b=%.4f\n",
                   step, loss->data[0], W->data[0], W->data[1], b->data[0]);
        }

        zf_tensor_release(loss);
    }

    int correct = 0;
    for (int n = 0; n < N; n++)
    {
        float z = X[n * 2 + 0] * W->data[0] + X[n * 2 + 1] * W->data[1] + b->data[0];
        float p = sigmoidf_stable(z);
        int pred = (p >= 0.5f) ? 1 : 0;
        int gt = (int)y[n];
        if (pred == gt)
            correct++;
    }

    printf("OR accuracy: %d/%d\n", correct, N);
    assert(correct == N);

    zf_tensor_release(targets);
    zf_tensor_release(logits);
    zf_tensor_release(b);
    zf_tensor_release(W);

    printf("[OK] train_linear_bce (OR) passed.\n");
    return 0;
}
