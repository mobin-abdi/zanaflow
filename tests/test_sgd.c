#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "zanaflow/tensor/tensor.h"
#include "zanaflow/core/parameter.h"
#include "zanaflow/optim/sgd.h"

int tensors_are_equal(Tensor *a, Tensor *b, float eps)
{
    if (!a || !b) return 0;
    if (a->ndim != b->ndim) return 0;
    if (a->size != b->size) return 0;

    for (int i = 0; i < a->ndim; i++)
    {
        if (a->shape[i] != b->shape[i]) return 0;
    }

    for (int i = 0; i < a->size; i++)
    {
        if (fabsf(a->data[i] - b->data[i]) > eps) return 0;
    }

    return 1;
}

int grad_is_equal_to_value(Tensor *t, float value, float eps)
{
    if (!t || !t->grad) return 0;

    for (int i = 0; i < t->size; i++)
    {
        if (fabsf(t->grad[i] - value) > eps) return 0;
    }

    return 1;
}

void test_sgd_create()
{
    printf("Running test_sgd_create...\n");

    int shape[] = {2, 2};
    Parameter *p1 = zf_parameter_create(shape, 2);
    Parameter *params[] = {p1};

    float learning_rate = 0.01f;
    SGD *opt = zf_sgd_create(params, 1, learning_rate);

    assert(opt != NULL);
    assert(opt->params != NULL);
    assert(opt->count == 1);
    assert(fabsf(opt->lr - learning_rate) < 1e-8f);

    printf("test_sgd_create PASSED.\n");

    zf_sgd_free(opt);
    zf_parameter_free(p1);
}

void test_sgd_step()
{
    printf("Running test_sgd_step...\n");

    int shape1[] = {2, 2};
    Parameter *p1 = zf_parameter_create(shape1, 2);
    assert(p1 != NULL);
    assert(p1->value != NULL);

    zf_tensor_fill(p1->value, 1.0f);
    assert(zf_tensor_ensure_grad(p1->value));
    for (int i = 0; i < p1->value->size; i++)
        p1->value->grad[i] = 0.5f;

    int shape2[] = {2};
    Parameter *p2 = zf_parameter_create(shape2, 1);
    assert(p2 != NULL);
    assert(p2->value != NULL);

    zf_tensor_fill(p2->value, 2.0f);
    assert(zf_tensor_ensure_grad(p2->value));
    for (int i = 0; i < p2->value->size; i++)
        p2->value->grad[i] = -0.1f;

    Parameter *params[2] = {p1, p2};

    float learning_rate = 0.1f;
    SGD *opt = zf_sgd_create(params, 2, learning_rate);
    assert(opt != NULL);

    zf_sgd_step(opt);

    int expected_w_shape[] = {2, 2};
    Tensor *expected_w = zf_tensor_create(expected_w_shape, 2);
    zf_tensor_fill(expected_w, 0.95f);

    int expected_b_shape[] = {2};
    Tensor *expected_b = zf_tensor_create(expected_b_shape, 1);
    zf_tensor_fill(expected_b, 2.01f);

    assert(tensors_are_equal(opt->params[0]->value, expected_w, 1e-6f));
    assert(tensors_are_equal(opt->params[1]->value, expected_b, 1e-6f));

    printf("test_sgd_step PASSED.\n");

    zf_sgd_free(opt);
    zf_parameter_free(p1);
    zf_parameter_free(p2);
    zf_tensor_release(expected_w);
    zf_tensor_release(expected_b);
}

void test_sgd_zero_grad()
{
    printf("Running test_sgd_zero_grad...\n");

    int shape1[] = {2};
    Parameter *p1 = zf_parameter_create(shape1, 1);
    assert(p1 != NULL);
    assert(p1->value != NULL);

    zf_tensor_fill(p1->value, 1.0f);
    assert(zf_tensor_ensure_grad(p1->value));
    for (int i = 0; i < p1->value->size; i++)
        p1->value->grad[i] = 0.5f;

    Parameter *params[1] = {p1};

    SGD *opt = zf_sgd_create(params, 1, 0.1f);
    assert(opt != NULL);

    zf_sgd_zero_grad(opt);

    assert(grad_is_equal_to_value(opt->params[0]->value, 0.0f, 1e-6f));

    printf("test_sgd_zero_grad PASSED.\n");

    zf_sgd_free(opt);
    zf_parameter_free(p1);
}

int main()
{
    test_sgd_create();
    test_sgd_step();
    test_sgd_zero_grad();

    printf("\nAll SGD tests passed successfully.\n");
    return 0;
}
