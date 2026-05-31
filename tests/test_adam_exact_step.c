#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <zanaflow/core/parameter.h>
#include <zanaflow/optim/adam.h>

static void assert_close(float got, float expected, float tol)
{
    float diff = fabsf(got - expected);
    if (diff > tol)
    {
        fprintf(stderr, "ASSERT_CLOSE failed: got=%g expected=%g diff=%g tol=%g\n",
                got, expected, diff, tol);
        assert(0);
    }
}

int main(void)
{
    int shape[1] = {1};
    Parameter *p = zf_parameter_create(shape, 1);
    assert(p && p->value && p->value->data && p->value->grad);

    // init
    p->value->data[0] = 1.0f;

    // grad
    float g = 2.0f;
    p->value->grad[0] = g;

    // Adam config
    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;

    Parameter *params[1] = {p};
    Adam *opt = zf_adam_create(params, 1, lr, beta1, beta2, eps);
    assert(opt);

    // For t=1 with bias-correction:
    // m_hat = g, v_hat = g^2
    float expected = 1.0f - lr * (g / (sqrtf(g * g) + eps));

    zf_adam_step(opt);

    float got = p->value->data[0];
    assert_close(got, expected, 1e-7f);

    zf_adam_free(opt);
    zf_parameter_free(p);

    printf("PASS: test_adam_exact_step\n");
    return 0;
}
