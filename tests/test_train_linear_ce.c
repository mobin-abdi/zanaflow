#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <zanaflow/tensor/tensor.h>
#include <zanaflow/autograd/autograd.h>
#include <zanaflow/loss/loss.h>

Tensor *zf_loss_cross_entropy(Tensor *logits, Tensor *target_indices);

static void zero_grad(Tensor *t)
{
    if (t && t->grad)
        memset(t->grad, 0, sizeof(float) * (size_t)t->size);
}

// forward linear: logits = XW + b
// X: [N,2], W:[2,C] row-major (w00,w01,w02,w10,w11,w12), b:[C] => logits:[N,C]
static void linear_forward_mc(const float *X, int N, const Tensor *W, const Tensor *b, int C, Tensor *logits)
{
    for (int n = 0; n < N; n++)
    {
        float x0 = X[n * 2 + 0];
        float x1 = X[n * 2 + 1];

        for (int c = 0; c < C; c++)
        {
            float w0c = W->data[0 * C + c];
            float w1c = W->data[1 * C + c];
            logits->data[n * C + c] = x0 * w0c + x1 * w1c + b->data[c];
        }
    }
}

static int argmax(const float *a, int C)
{
    int mi = 0;
    float mv = a[0];
    for (int i = 1; i < C; i++)
        if (a[i] > mv)
        {
            mv = a[i];
            mi = i;
        }
    return mi;
}

int main(void)
{
    // 3 clusters in 2D
    // class 0 around (-2, -2), class 1 around (2, -2), class 2 around (0, 2)
    // N=9 (3 samples per class)
    float X[18] = {
        -2.0f, -2.0f,
        -2.5f, -1.5f,
        -1.5f, -2.3f,

        2.0f, -2.0f,
        2.4f, -1.7f,
        1.6f, -2.4f,

        0.0f, 2.0f,
        0.4f, 2.5f,
        -0.3f, 1.6f};
    int y[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    int N = 9;
    int C = 3;

    // params: W[2,C] and b[C]
    int wshape[2] = {2, C};
    Tensor *W = zf_tensor_create(wshape, 2);
    assert(W && W->data);
    W->requires_grad = 1;

    int bshape[1] = {C};
    Tensor *b = zf_tensor_create(bshape, 1);
    assert(b && b->data);
    b->requires_grad = 1;

    // init small random-ish (deterministic)
    for (int i = 0; i < W->size; i++)
        W->data[i] = 0.01f * (float)(i - 2);
    for (int i = 0; i < b->size; i++)
        b->data[i] = 0.0f;

    // logits tensor [N,C]
    int lshape[2] = {N, C};
    Tensor *logits = zf_tensor_create(lshape, 2);
    assert(logits && logits->data);
    logits->requires_grad = 1;

    // targets indices tensor [N]
    int tshape[1] = {N};
    Tensor *tidx = zf_tensor_create(tshape, 1);
    assert(tidx && tidx->data);
    tidx->requires_grad = 0;
    for (int n = 0; n < N; n++)
        tidx->data[n] = (float)y[n];

    float lr = 0.2f;
    int steps = 1500;

    for (int step = 0; step < steps; step++)
    {
        // forward
        linear_forward_mc(X, N, W, b, C, logits);

        // zero grads
        zero_grad(logits);
        zero_grad(W);
        zero_grad(b);

        // loss
        Tensor *loss = zf_loss_cross_entropy(logits, tidx);
        assert(loss && loss->data);

        // backward gives dL/dlogits into logits->grad
        zf_backward(loss);

        // manual backprop through linear to accumulate into dW, db
        // logits->grad already includes mean reduction of CE over N (if you implemented mean).
        // dW = X^T * dZ , db = sum_n dZ
        float dW[2 * 3] = {0};
        float dbv[3] = {0};

        for (int n = 0; n < N; n++)
        {
            float x0 = X[n * 2 + 0];
            float x1 = X[n * 2 + 1];

            for (int c = 0; c < C; c++)
            {
                float dz = logits->grad[n * C + c];
                dW[0 * C + c] += x0 * dz;
                dW[1 * C + c] += x1 * dz;
                dbv[c] += dz;
            }
        }

        // SGD update
        for (int i = 0; i < 2 * C; i++)
            W->data[i] -= lr * dW[i];
        for (int c = 0; c < C; c++)
            b->data[c] -= lr * dbv[c];

        if (step % 200 == 0 || step == steps - 1)
        {
            printf("step=%d loss=%.6f\n", step, loss->data[0]);
        }

        zf_tensor_release(loss);
    }

    // evaluate
    int correct = 0;
    for (int n = 0; n < N; n++)
    {
        float z[3];
        float x0 = X[n * 2 + 0];
        float x1 = X[n * 2 + 1];
        for (int c = 0; c < C; c++)
            z[c] = x0 * W->data[0 * C + c] + x1 * W->data[1 * C + c] + b->data[c];

        int pred = argmax(z, C);
        if (pred == y[n])
            correct++;
    }

    printf("3-class accuracy: %d/%d\n", correct, N);
    assert(correct >= 8); // should reach near perfect, tolerate 1 mistake

    zf_tensor_release(tidx);
    zf_tensor_release(logits);
    zf_tensor_release(b);
    zf_tensor_release(W);

    printf("[OK] train_linear_ce (3-class) passed.\n");
    return 0;
}
