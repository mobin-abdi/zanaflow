#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <zanaflow.h>
#include <zanaflow/autograd/autograd.h>
#include <zanaflow/nn/dense.h>
#include <zanaflow/optim/adam.h>

static float frand_uniform(float a, float b)
{
    return a + (b - a) * (float)rand() / (float)RAND_MAX;
}

static void fill_dataset_sin(Tensor *X, Tensor *Y)
{
    if (!X || !Y)
    {
        return;
    }
    if (X->ndim != 2 || Y->ndim != 2)
    {
        return;
    }
    if (X->shape[0] != Y->shape[0])
    {
        return;
    }
    if (X->shape[1] != 1 || Y->shape[1] != 1)
    {
        return;
    }

    int n = X->shape[0];
    for (int i = 0; i < n; i++)
    {
        float x = frand_uniform(-3.1415926f, 3.1415926f);
        X->data[i] = x;
        Y->data[i] = sinf(x);
    }
}

static float compute_mse_raw(Tensor *pred, Tensor *target)
{
    if (!pred || !target)
    {
        return 0.0f;
    }
    if (pred->size != target->size)
    {
        return 0.0f;
    }
    float acc = 0.0f;
    for (int i = 0; i < pred->size; i++)
    {
        float d = pred->data[i] - target->data[i];
        acc += d * d;
    }
    return acc / (float)pred->size;
}

static void print_param_stats(const char *name, Parameter *p)
{
    if (!p || !p->value)
    {
        printf("%s: <null>\n", name);
        return;
    }
    Tensor *t = p->value;
    float v0 = (t->data && t->size > 0) ? t->data[0] : 0.0f;
    float g0 = (t->grad && t->size > 0) ? t->grad[0] : 0.0f;
    printf("%s: req=%d size=%d v0=%.6f g0=%.6f\n", name, t->requires_grad, t->size, v0, g0);
}

int main(void)
{
    srand(0);

    const int N = 128;
    int x_shape[] = {N, 1};
    int y_shape[] = {N, 1};

    Tensor *X = zf_tensor_create(x_shape, 2);
    Tensor *Y = zf_tensor_create(y_shape, 2);
    if (!X || !Y)
    {
        fprintf(stderr, "failed to create dataset tensors\n");
        return 1;
    }
    X->requires_grad = 0;
    Y->requires_grad = 0;
    fill_dataset_sin(X, Y);

    DenseLayer *fc1 = zf_dense_create(1, 16);
    DenseLayer *fc2 = zf_dense_create(16, 1);

    if (!fc1 || !fc2)
    {
        fprintf(stderr, "failed to create dense layers\n");
        zf_tensor_release(X);
        zf_tensor_release(Y);
        if (fc1)
        {
            zf_dense_free(fc1);
        }
        if (fc2)
        {
            zf_dense_free(fc2);
        }
        return 1;
    }

    printf("Initial parameter flags:\n");
    print_param_stats("fc1.W", fc1->weights);
    print_param_stats("fc1.b", fc1->bias);
    print_param_stats("fc2.W", fc2->weights);
    print_param_stats("fc2.b", fc2->bias);

    zf_init_he_uniform(fc1->weights->value, fc1->bias->value, fc1->in_features);
    zf_init_he_uniform(fc2->weights->value, fc2->bias->value, fc2->in_features);

    Parameter *params[4] = {0};
    if (zf_dense_parameters(fc1, &params[0]) != 2 ||
        zf_dense_parameters(fc2, &params[2]) != 2)
    {
        fprintf(stderr, "failed to collect parameters\n");
        zf_dense_free(fc1);
        zf_dense_free(fc2);
        zf_tensor_release(X);
        zf_tensor_release(Y);
        return 1;
    }

    Adam *opt = zf_adam_create(params, 4, 0.001f, 0.9f, 0.999f, 1e-2f);
    if (!opt)
    {
        fprintf(stderr, "failed to create optimizer\n");
        zf_dense_free(fc1);
        zf_dense_free(fc2);
        zf_tensor_release(X);
        zf_tensor_release(Y);
        return 1;
    }

    const int epochs = 2000;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        zf_adam_zero_grad(opt);

        Tensor *h1 = zf_dense_forward(fc1, X);
        if (!h1)
        {
            fprintf(stderr, "h1 forward failed at epoch %d\n", epoch);
            break;
        }

        Tensor *a1 = zf_tanh(h1);
        if (!a1)
        {
            fprintf(stderr, "tanh failed at epoch %d\n", epoch);
            zf_tensor_release(h1);
            break;
        }

        Tensor *pred = zf_dense_forward(fc2, a1);
        if (!pred)
        {
            fprintf(stderr, "pred forward failed at epoch %d\n", epoch);
            zf_tensor_release(a1);
            zf_tensor_release(h1);
            break;
        }

        Tensor *loss = zf_loss_mse(pred, Y);
        if (!loss)
        {
            fprintf(stderr, "loss failed at epoch %d\n", epoch);
            zf_tensor_release(pred);
            zf_tensor_release(a1);
            zf_tensor_release(h1);
            break;
        }

        if (epoch == 0)
        {
            printf("Sanity: pred->requires_grad=%d loss->requires_grad=%d\n", pred->requires_grad, loss->requires_grad);
        }

        zf_backward(loss);

        if (epoch % 200 == 0)
        {
            float mse = compute_mse_raw(pred, Y);
            printf("Epoch %4d | loss=%.6f | mse=%.6f\n", epoch, (loss->data ? loss->data[0] : 0.0f), mse);
            print_param_stats("fc1.W", fc1->weights);
            print_param_stats("fc1.b", fc1->bias);
            print_param_stats("fc2.W", fc2->weights);
            print_param_stats("fc2.b", fc2->bias);
        }

        zf_adam_step(opt);

        zf_tensor_release(loss);
        zf_tensor_release(pred);
        zf_tensor_release(a1);
        zf_tensor_release(h1);
    }

    zf_adam_free(opt);
    zf_dense_free(fc1);
    zf_dense_free(fc2);
    zf_tensor_release(X);
    zf_tensor_release(Y);

    printf("Training finished.\n");
    return 0;
}
