#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <zanaflow/tensor/tensor.h>
#include <zanaflow/ops/ops.h>
#include <zanaflow/autograd/autograd.h>

static Tensor *compute_loss(Tensor *x)
{
    if (!x)
    {
        fprintf(stderr, "compute_loss: x is NULL\n");
        return NULL;
    }

    // loss = sum(x*x)
    Tensor *sq = zf_tensor_mul(x, x);
    if (!sq)
    {
        fprintf(stderr, "compute_loss: zf_tensor_mul failed\n");
        return NULL;
    }

    Tensor *loss = zf_tensor_sum_all(sq);
    zf_tensor_release(sq);

    if (!loss)
    {
        fprintf(stderr, "compute_loss: zf_tensor_sum_all failed\n");
        return NULL;
    }

    return loss;
}

static void numerical_gradient(Tensor *x, Tensor *(*loss_fn)(Tensor *), float *num_grad)
{
    const double eps = 1e-3;

    for (int i = 0; i < x->size; i++)
    {
        double orig = (double)x->data[i];

        x->data[i] = (float)(orig + eps);
        Tensor *loss1 = loss_fn(x);
        double f1 = loss1 ? (double)loss1->data[0] : 0.0;
        if (loss1) zf_tensor_release(loss1);

        x->data[i] = (float)(orig - eps);
        Tensor *loss2 = loss_fn(x);
        double f2 = loss2 ? (double)loss2->data[0] : 0.0;
        if (loss2) zf_tensor_release(loss2);

        num_grad[i] = (float)((f1 - f2) / (2.0 * eps));
        x->data[i] = (float)orig;
    }
}

int main(void)
{
    int shape[] = {4};
    float vals[] = {1.0f, 2.0f, 3.0f, 4.0f};

    Tensor *x = zf_tensor_create(shape, 1);
    if (!x)
    {
        fprintf(stderr, "Failed to create x\n");
        return 1;
    }

    x->requires_grad = 1;
    for (int i = 0; i < 4; i++)
        x->data[i] = vals[i];

    // loss forward
    Tensor *loss = compute_loss(x);
    if (!loss)
    {
        zf_tensor_release(x);
        return 1;
    }

    // backward (با API جدید: خود zf_backward روی loss grad=1 می‌گذارد)
    zf_backward(loss);

    // numerical grad
    float *num_grad = (float*)malloc((size_t)x->size * sizeof(float));
    if (!num_grad)
    {
        fprintf(stderr, "Failed to alloc num_grad\n");
        zf_tensor_release(loss);
        zf_tensor_release(x);
        return 1;
    }

    numerical_gradient(x, compute_loss, num_grad);

    printf("Index | AutoGrad     | Numerical    | Diff\n");
    int passed = 1;
    for (int i = 0; i < x->size; i++)
    {
        float auto_g = x->grad ? x->grad[i] : 0.0f;
        float diff = fabsf(auto_g - num_grad[i]);
        printf("%5d | %11.6f | %11.6f | %e\n", i, auto_g, num_grad[i], diff);
        if (diff > 1e-2f) passed = 0;
    }

    printf(passed ? "\nGradient check PASSED\n" : "\nGradient check FAILED\n");

    free(num_grad);
    zf_tensor_release(loss);
    zf_tensor_release(x);
    return passed ? 0 : 1;
}
