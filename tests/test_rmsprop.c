#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <zanaflow/core/parameter.h>
#include <zanaflow/tensor/tensor.h>
#include <zanaflow/optim/rmsprop.h>

static int almost_equal(float a, float b, float tol)
{
    return fabsf(a - b) <= tol;
}

int main(void)
{
    {
        int shape[1] = {1};
        Parameter *p = zf_parameter_create(shape, 1);
        assert(p != NULL);
        assert(p->value != NULL);

        p->value->requires_grad = 1;
        zf_tensor_ensure_grad(p->value);

        p->value->data[0] = 1.0f;
        p->value->grad[0] = 2.0f;

        Parameter *params[1] = {p};
        RMSProp *opt = zf_rmsprop_create(params, 1, 0.1f, 0.0f, 1e-8f);
        assert(opt != NULL);

        int rc = zf_rmsprop_step(opt);
        assert(rc == 0);

        float expected = 1.0f - 0.1f * (2.0f / (2.0f + 1e-8f));
        assert(almost_equal(p->value->data[0], expected, 1e-6f));

        zf_rmsprop_free(opt);
        zf_parameter_free(p);
    }

    {
        int shape[1] = {1};
        Parameter *p = zf_parameter_create(shape, 1);
        assert(p != NULL);
        assert(p->value != NULL);

        p->value->requires_grad = 1;
        zf_tensor_ensure_grad(p->value);

        p->value->data[0] = 1.0f;

        Parameter *params[1] = {p};
        RMSProp *opt = zf_rmsprop_create(params, 1, 0.1f, 0.9f, 1e-8f);
        assert(opt != NULL);

        p->value->grad[0] = 1.0f;
        float before1 = p->value->data[0];
        assert(zf_rmsprop_step(opt) == 0);
        float after1 = p->value->data[0];

        p->value->grad[0] = 1.0f;
        float before2 = p->value->data[0];
        assert(zf_rmsprop_step(opt) == 0);
        float after2 = p->value->data[0];

        float delta1 = before1 - after1;
        float delta2 = before2 - after2;

        assert(delta1 > 0.0f);
        assert(delta2 > 0.0f);
        assert(delta2 < delta1);

        zf_rmsprop_free(opt);
        zf_parameter_free(p);
    }

    printf("[OK] test_rmsprop passed.\n");
    return 0;
}
