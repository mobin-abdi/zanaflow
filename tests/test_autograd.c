#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <zanaflow/tensor/tensor.h>
#include <zanaflow/ops/ops.h>
#include <zanaflow/autograd/autograd.h>

Tensor *compute_loss(Tensor *x) {
    if (!x || !x->requires_grad) {
        fprintf(stderr, "Error in compute_loss: input must exist and require gradients.\n");
        return NULL;
    }

    Tensor *sq = tensor_mul(x, x);
    if (!sq) {
        fprintf(stderr, "Error: tensor_mul failed in compute_loss.\n");
        return NULL;
    }

    Tensor *loss = tensor_sum_all(sq);
    if (!loss) {
        fprintf(stderr, "Error: tensor_sum_all failed in compute_loss.\n");
        tensor_release(sq);
        return NULL;
    }

    tensor_release(sq);
    return loss;
}

void numerical_gradient(Tensor *x, Tensor *(*loss_fn)(Tensor *), float *num_grad) {
    if (!x || !loss_fn || !num_grad) return;

    const double eps = 1e-3;  // برای float بهتر از 1e-4
    for (int i = 0; i < x->size; i++) {
        double orig = (double)x->data[i];

        x->data[i] = (float)(orig + eps);
        Tensor *loss1 = loss_fn(x);
        if (!loss1) {
            fprintf(stderr, "Error computing loss1 in numerical_gradient.\n");
            x->data[i] = (float)orig;
            return;
        }
        double f1 = (double)loss1->data[0];
        tensor_release(loss1);

        x->data[i] = (float)(orig - eps);
        Tensor *loss2 = loss_fn(x);
        if (!loss2) {
            fprintf(stderr, "Error computing loss2 in numerical_gradient.\n");
            x->data[i] = (float)orig;
            return;
        }
        double f2 = (double)loss2->data[0];
        tensor_release(loss2);

        num_grad[i] = (float)((f1 - f2) / (2.0 * eps));
        x->data[i] = (float)orig;
    }
}

int main() {
    int shape[] = {4};
    float vals[] = {1.0f, 2.0f, 3.0f, 4.0f};

    Tensor *x = tensor_create(shape, 1);
    if (!x) {
        fprintf(stderr, "Failed to create tensor x.\n");
        return 1;
    }

    x->requires_grad = 1;
    for (int i = 0; i < 4; i++) x->data[i] = vals[i];

    tensor_zero_grad(x);

    Tensor *loss = compute_loss(x);
    if (!loss) {
        tensor_release(x);
        return 1;
    }

    // seed gradient for scalar loss
    tensor_ensure_grad(loss);
    loss->grad[0] = 1.0f;

    zanaflow_backward(loss);

    float *num_grad = (float *)malloc(x->size * sizeof(float));
    if (!num_grad) {
        fprintf(stderr, "Failed to allocate num_grad.\n");
        tensor_release(loss);
        tensor_release(x);
        return 1;
    }

    numerical_gradient(x, compute_loss, num_grad);

    printf("Index | AutoGrad   | Numerical  | Diff\n");
    int passed = 1;
    for (int i = 0; i < x->size; i++) {
        float auto_g = x->grad ? x->grad[i] : 0.0f;
        float diff = fabsf(auto_g - num_grad[i]);
        printf("%5d | %10.6f | %10.6f | %e\n", i, auto_g, num_grad[i], diff);
        if (diff > 1e-2f) passed = 0;
    }

    printf(passed ? "\nGradient check PASSED!\n" : "\nGradient check FAILED!\n");

    free(num_grad);
    tensor_release(loss);
    tensor_release(x);
    return passed ? 0 : 1;
}
