#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <zanaflow/core/parameter.h>
#include <zanaflow/optim/adam.h>

static int float_ne(float a, float b)
{
    return fabsf(a - b) > 1e-12f;
}

int main(void)
{
    int shape[1] = {1};
    Parameter *p = zf_parameter_create(shape, 1);
    assert(p && p->value && p->value->data && p->value->grad);

    p->value->data[0] = 1.0f;
    p->value->grad[0] = 0.5f;

    Parameter *params[1] = {p};
    Adam *opt = zf_adam_create(params, 1, 0.01f, 0.9f, 0.999f, 1e-8f);
    assert(opt);

    float before = p->value->data[0];
    zf_adam_step(opt);
    float after = p->value->data[0];

    assert(float_ne(before, after));
    assert(after < before);

    zf_adam_free(opt);
    zf_parameter_free(p);

    printf("PASS: test_adam_sanity\n");
    return 0;
}
