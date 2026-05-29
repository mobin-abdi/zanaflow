#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <zanaflow/core/parameter.h>
#include <zanaflow/optim/sgd.h>
#include <zanaflow/tensor/tensor.h>

Tensor *mock_tensor_add(const Tensor *a, const Tensor *b) {
    if (!a || !b || a->size != b->size)
    {
        return NULL;
    }

    Tensor *out = tensor_create(a->shape, a->ndim);
    if (!out) 
    {
        return NULL;
    }

    for (int i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    
    return out;
}

Tensor *mock_tensor_mul_scalar(const Tensor *a, float s) {
    if (!a)
    {
        return NULL;
    }

    Tensor *out = tensor_create(a->shape, a->ndim);
    if (!out){
        return NULL;
    }

    for (int i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] * s;
    }

    return out;
}

Tensor *mock_tensor_clone(const Tensor *t) {
    if (t == NULL)
    {
        return NULL;
    }

    Tensor *out = tensor_create(t->shape, t->ndim);
    if (!out)
    {
        return NULL;
    }

    for (int i = 0; i < t->size; i++)
    {
        out->data[i] = t->data[i];
    }
    
    return out;
}

#define tensor_add mock_tensor_add
#define tensor_mul_scalar mock_tensor_mul_scalar
#define tensor_clone mock_tensor_clone

int tensors_are_equal(const Tensor *a, const Tensor *b, float epsilon) {
    if (!a || !b)
    {
        return 0;
    }

    if (a->size != b->size)
    {
        return 0;
    }

    for (int i = 0; i < a->size; i++) {
        if (fabs(a->data[i] - b->data[i]) > epsilon) {
            return 0;
        }
    }

    return 1;
}

void test_sgd_step() {
    printf("Running test_sgd_step...\n");

    int shape1[] = {2, 2};
    Tensor *w_val = tensor_create(shape1, 2); tensor_fill(w_val, 1.0f);
    Tensor *w_grad = tensor_create(shape1, 2); tensor_fill(w_grad, 0.5f);
    Parameter p1 = {w_val, w_grad};

    int shape2[] = {2};
    Tensor *b_val = tensor_create(shape2, 1); tensor_fill(b_val, 2.0f);
    Tensor *b_grad = tensor_create(shape2, 1); tensor_fill(b_grad, -0.1f);
    Parameter p2 = {b_val, b_grad};

    Parameter params[2] = {p1, p2};
    float learning_rate = 0.1f;
    SGD *opt = sgd_create(params, 2, learning_rate);
    assert(opt != NULL);

    sgd_step(opt);

    Tensor *expected_w_data = tensor_create(shape1, 2); tensor_fill(expected_w_data, 0.95f);
    Tensor *expected_b_data = tensor_create(shape2, 1); tensor_fill(expected_b_data, 2.01f);

    assert(tensors_are_equal(opt->params[0].value, expected_w_data, 1e-6));
    assert(tensors_are_equal(opt->params[1].value, expected_b_data, 1e-6));

    printf("test_sgd_step PASSED.\n");

    sgd_free(opt);
    tensor_free(expected_w_data);
    tensor_free(expected_b_data);
    tensor_free(w_val); tensor_free(w_grad);
    tensor_free(b_val); tensor_free(b_grad);
}

void test_sgd_zero_grad() {
    printf("Running test_sgd_zero_grad...\n");

    int shape1[] = {2};
    Tensor *w_val = tensor_create(shape1, 1); tensor_fill(w_val, 1.0f);
    Tensor *w_grad = tensor_create(shape1, 1); tensor_fill(w_grad, 0.5f); // Non-zero grad
    Parameter p1 = {w_val, w_grad};
    Parameter params[1] = {p1};
    SGD *opt = sgd_create(params, 1, 0.1f);
    assert(opt != NULL);

    sgd_zero_grad(opt);

    Tensor *expected_grad = tensor_create(shape1, 1); tensor_fill(expected_grad, 0.0f);
    assert(tensors_are_equal(opt->params[0].grad, expected_grad, 1e-6));

    assert(tensors_are_equal(opt->params[0].value, w_val, 1e-6));

    printf("test_sgd_zero_grad PASSED.\n");

    sgd_free(opt);
    tensor_free(expected_grad);
    tensor_free(w_val); tensor_free(w_grad);
}

int main() {
    test_sgd_step();
    test_sgd_zero_grad();

    printf("\nAll Optimizer tests finished.\n");
    return 0;
}
