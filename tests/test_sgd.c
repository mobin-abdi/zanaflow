#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <zanaflow/core/parameter.h>
#include <zanaflow/optim/sgd.h>
#include <zanaflow/tensor/tensor.h>

int tensors_are_equal(const Tensor *a, const Tensor *b, float epsilon)
{
    if (!a || !b)
    {
        return 0;
    }

    if (a->size != b->size)
    {
        return 0;
    }

    for (int i = 0; i < a->size; i++)
    {
        if (fabs(a->data[i] - b->data[i]) > epsilon)
        {
            return 0;
        }
    }

    return 1;
}

int grad_is_equal_to_value(const Tensor *t, float value, float epsilon)
{
    if (!t || !t->grad)
    {
        return 0;
    }

    for (int i = 0; i < t->size; i++)
    {
        if (fabs(t->grad[i] - value) > epsilon)
        {
            return 0;
        }
    }

    return 1;
}

void test_sgd_step()
{
    printf("Running test_sgd_step...\n");

    int shape1[] = {2, 2};
    Tensor *w_val = zf_tensor_create(shape1, 2);
    zf_tensor_fill(w_val, 1.0f);
    assert(zf_tensor_ensure_grad(w_val));
    for (int i = 0; i < w_val->size; i++)
    {
        w_val->grad[i] = 0.5f;
    }
    Parameter p1 = {w_val};

    int shape2[] = {2};
    Tensor *b_val = zf_tensor_create(shape2, 1);
    zf_tensor_fill(b_val, 2.0f);
    assert(zf_tensor_ensure_grad(b_val));
    for (int i = 0; i < b_val->size; i++)
    {
        b_val->grad[i] = -0.1f;
    }
    Parameter p2 = {b_val};

    Parameter params[2] = {p1, p2};
    float learning_rate = 0.1f;
    SGD *opt = zf_sgd_create(params, 2, learning_rate);
    assert(opt != NULL);

    zf_sgd_step(opt);

    Tensor *expected_w_data = zf_tensor_create(shape1, 2);
    zf_tensor_fill(expected_w_data, 0.95f);

    Tensor *expected_b_data = zf_tensor_create(shape2, 1);
    zf_tensor_fill(expected_b_data, 2.01f);

    assert(tensors_are_equal(opt->params[0].value, expected_w_data, 1e-6f));
    assert(tensors_are_equal(opt->params[1].value, expected_b_data, 1e-6f));

    printf("test_sgd_step PASSED.\n");

    zf_sgd_free(opt);
    zf_tensor_release(expected_w_data);
    zf_tensor_release(expected_b_data);
    zf_tensor_release(w_val);
    zf_tensor_release(b_val);
}

void test_sgd_zero_grad()
{
    printf("Running test_sgd_zero_grad...\n");

    int shape1[] = {2};
    Tensor *w_val = zf_tensor_create(shape1, 1);
    zf_tensor_fill(w_val, 1.0f);
    assert(zf_tensor_ensure_grad(w_val));
    for (int i = 0; i < w_val->size; i++)
    {
        w_val->grad[i] = 0.5f;
    }

    Parameter p1 = {w_val};
    Parameter params[1] = {p1};

    SGD *opt = zf_sgd_create(params, 1, 0.1f);
    assert(opt != NULL);

    zf_sgd_zero_grad(opt);

    assert(grad_is_equal_to_value(opt->params[0].value, 0.0f, 1e-6f));
    assert(tensors_are_equal(opt->params[0].value, w_val, 1e-6f));

    printf("test_sgd_zero_grad PASSED.\n");

    zf_sgd_free(opt);
    zf_tensor_release(w_val);
}

int main()
{
    test_sgd_step();
    test_sgd_zero_grad();

    printf("\nAll Optimizer tests finished.\n");
    return 0;
}
