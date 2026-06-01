#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <zanaflow/tensor/tensor.h>
#include <zanaflow/autograd/autograd.h>
#include <zanaflow/loss/loss.h>

Tensor *zf_loss_cross_entropy(Tensor *logits, Tensor *target_indices);
Tensor *zf_loss_binary_cross_entropy(Tensor *logits, Tensor *targets);

static float rel_error(float a, float b)
{
    float denom = fmaxf(1.0f, fabsf(a) + fabsf(b));
    return fabsf(a - b) / denom;
}

static float fd_grad_ce_at(Tensor *logits, Tensor *tidx, int idx, float eps)
{
    float old = logits->data[idx];

    logits->data[idx] = old + eps;
    Tensor *lp = zf_loss_cross_entropy(logits, tidx);
    assert(lp && lp->data);
    float fp = lp->data[0];
    zf_tensor_release(lp);

    logits->data[idx] = old - eps;
    Tensor *lm = zf_loss_cross_entropy(logits, tidx);
    assert(lm && lm->data);
    float fm = lm->data[0];
    zf_tensor_release(lm);

    logits->data[idx] = old;
    return (fp - fm) / (2.0f * eps);
}

static float fd_grad_bce_at(Tensor *logits, Tensor *targets, int idx, float eps)
{
    float old = logits->data[idx];

    logits->data[idx] = old + eps;
    Tensor *lp = zf_loss_binary_cross_entropy(logits, targets);
    assert(lp && lp->data);
    float fp = lp->data[0];
    zf_tensor_release(lp);

    logits->data[idx] = old - eps;
    Tensor *lm = zf_loss_binary_cross_entropy(logits, targets);
    assert(lm && lm->data);
    float fm = lm->data[0];
    zf_tensor_release(lm);

    logits->data[idx] = old;
    return (fp - fm) / (2.0f * eps);
}

static void test_cross_entropy()
{
    // logits: [2,3]
    int lshape[2] = {2, 3};
    Tensor *logits = zf_tensor_create(lshape, 2);
    assert(logits && logits->data);
    logits->requires_grad = 1;

    float ldata[6] = {
        2.0f, 1.0f, 0.1f,
        0.5f, 2.5f, 0.3f};
    memcpy(logits->data, ldata, sizeof(ldata));

    // target indices: [2]
    int tshape[1] = {2};
    Tensor *tidx = zf_tensor_create(tshape, 1);
    assert(tidx && tidx->data);
    // stored as float in your tensors -> cast to int inside CE
    tidx->data[0] = 0.0f;
    tidx->data[1] = 1.0f;

    Tensor *loss = zf_loss_cross_entropy(logits, tidx);
    assert(loss && loss->data);
    assert(loss->data[0] > 0.0f);

    zf_backward(loss);
    assert(logits->grad != NULL);

    // finite-difference checks
    float eps = 1e-3f;
    int check_ids[] = {0, 1, 2, 3, 4, 5}; // all elements
    int K = (int)(sizeof(check_ids) / sizeof(check_ids[0]));

    float max_rel = 0.0f;
    for (int k = 0; k < K; k++)
    {
        int idx = check_ids[k];
        float fd = fd_grad_ce_at(logits, tidx, idx, eps);
        float ad = logits->grad[idx];
        float r = rel_error(fd, ad);
        if (r > max_rel)
            max_rel = r;

        // CE should be smooth; this tolerance is usually fine for eps=1e-3
        if (r > 5e-2f)
        {
            fprintf(stderr,
                    "CE gradcheck FAIL idx=%d fd=%+.6f ad=%+.6f rel=%.6f\n",
                    idx, fd, ad, r);
            assert(0);
        }
    }

    printf("[OK] CrossEntropy forward/backward (max rel err = %.6f)\n", max_rel);

    zf_tensor_release(loss);
    zf_tensor_release(tidx);
    zf_tensor_release(logits);
}

static void test_bce_with_logits()
{
    // logits/targets: [6]
    int shape[1] = {6};
    Tensor *logits = zf_tensor_create(shape, 1);
    Tensor *targets = zf_tensor_create(shape, 1);
    assert(logits && targets);
    assert(logits->data && targets->data);

    logits->requires_grad = 1;

    // mix of easy/harder values for stability
    float zdata[6] = {-5.0f, -1.5f, -0.1f, 0.2f, 2.0f, 6.0f};
    float ydata[6] = {0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f};
    memcpy(logits->data, zdata, sizeof(zdata));
    memcpy(targets->data, ydata, sizeof(ydata));

    Tensor *loss = zf_loss_binary_cross_entropy(logits, targets);
    assert(loss && loss->data);
    assert(loss->data[0] > 0.0f);

    zf_backward(loss);
    assert(logits->grad != NULL);

    float eps = 1e-3f;
    float max_rel = 0.0f;

    for (int idx = 0; idx < logits->size; idx++)
    {
        float fd = fd_grad_bce_at(logits, targets, idx, eps);
        float ad = logits->grad[idx];
        float r = rel_error(fd, ad);
        if (r > max_rel)
            max_rel = r;

        // BCE also smooth; tolerance similar
        if (r > 5e-2f)
        {
            fprintf(stderr,
                    "BCEWithLogits gradcheck FAIL idx=%d fd=%+.6f ad=%+.6f rel=%.6f\n",
                    idx, fd, ad, r);
            assert(0);
        }
    }

    printf("[OK] BCEWithLogits forward/backward (max rel err = %.6f)\n", max_rel);

    zf_tensor_release(loss);
    zf_tensor_release(targets);
    zf_tensor_release(logits);
}

int main(void)
{
    test_cross_entropy();
    test_bce_with_logits();
    printf("All loss tests passed.\n");
    return 0;
}
