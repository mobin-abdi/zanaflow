#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdbool.h>
#include <zanaflow/tensor/tensor.h>
#include <zanaflow/ops/activations.h>
#include <zanaflow/autograd/autograd.h>
#include <zanaflow/ops/ops.h>

static int almost_equal(float a, float b, float eps)
{
    return fabsf(a - b) < eps;
}

void test_leaky_relu_forward()
{
    int shape[] = {4};
    Tensor *t = zf_tensor_create(shape, 1);
    assert(t);

    t->data[0] = -2.0f;
    t->data[1] = 0.0f;
    t->data[2] = 3.0f;
    t->data[3] = -1.0f;

    Tensor *y = zf_leaky_relu(t, 0.1f);
    assert(y);

    assert(almost_equal(y->data[0], -0.2f, 1e-6f));
    assert(almost_equal(y->data[1], 0.0f, 1e-6f));
    assert(almost_equal(y->data[2], 3.0f, 1e-6f));
    assert(almost_equal(y->data[3], -0.1f, 1e-6f));

    zf_tensor_release(t);
    zf_tensor_release(y);
    printf("Test LeakyReLU Forward: PASSED\n");
}

void test_tanh_forward()
{
    int shape[] = {3};
    Tensor *t = zf_tensor_create(shape, 1);
    assert(t);

    t->data[0] = 0.0f;
    t->data[1] = 1.0f;
    t->data[2] = -1.0f;

    Tensor *y = zf_tanh(t);
    assert(y);

    assert(almost_equal(y->data[0], 0.0f, 1e-6f));
    assert(almost_equal(y->data[1], tanhf(1.0f), 1e-6f));
    assert(almost_equal(y->data[2], tanhf(-1.0f), 1e-6f));

    zf_tensor_release(t);
    zf_tensor_release(y);
    printf("Test Tanh Forward: PASSED\n");
}

void test_softmax_1d_forward()
{
    int shape[] = {3};
    Tensor *t = zf_tensor_create(shape, 1);
    assert(t);

    t->data[0] = 1.0f;
    t->data[1] = 2.0f;
    t->data[2] = 3.0f;

    Tensor *y = zf_softmax(t);
    assert(y);

    float sum = y->data[0] + y->data[1] + y->data[2];
    assert(almost_equal(sum, 1.0f, 1e-5f));

    assert(y->data[2] > y->data[1]);
    assert(y->data[1] > y->data[0]);

    zf_tensor_release(t);
    zf_tensor_release(y);
    printf("Test Softmax 1D Forward: PASSED\n");
}

void test_tanh_autograd_graph()
{
    int shape[] = {2};
    Tensor *t = zf_tensor_create(shape, 1);
    assert(t);

    t->requires_grad = true;
    t->data[0] = 0.5f;
    t->data[1] = -1.0f;

    Tensor *y = zf_tanh(t);
    assert(y);

    assert(y->requires_grad == true);
    assert(y->grad_node != NULL);
    assert(y->grad_node->inputs[0] == t);
    assert(y->grad_node->output == y);

    zf_tensor_release(t);
    zf_tensor_release(y);
    printf("Test Tanh Autograd Graph: PASSED\n");
}

void test_leaky_relu_autograd_graph()
{
    int shape[] = {2};
    Tensor *t = zf_tensor_create(shape, 1);
    assert(t);

    t->requires_grad = true;
    t->data[0] = -2.0f;
    t->data[1] = 3.0f;

    Tensor *y = zf_leaky_relu(t, 0.1f);
    assert(y);

    assert(y->requires_grad == true);
    assert(y->grad_node != NULL);
    assert(y->grad_node->inputs[0] == t);
    assert(y->grad_node->output == y);
    assert(y->grad_node->ctx != NULL);

    float alpha = *((float *)y->grad_node->ctx);
    assert(almost_equal(alpha, 0.1f, 1e-6f));

    zf_tensor_release(t);
    zf_tensor_release(y);
    printf("Test LeakyReLU Autograd Graph: PASSED\n");
}

void test_softmax_autograd_graph()
{
    int shape[] = {3};
    Tensor *t = zf_tensor_create(shape, 1);
    assert(t);

    t->requires_grad = true;
    t->data[0] = 1.0f;
    t->data[1] = 2.0f;
    t->data[2] = 3.0f;

    Tensor *y = zf_softmax(t);
    assert(y);

    assert(y->requires_grad == true);
    assert(y->grad_node != NULL);
    assert(y->grad_node->inputs[0] == t);
    assert(y->grad_node->output == y);

    zf_tensor_release(t);
    zf_tensor_release(y);
    printf("Test Softmax Autograd Graph: PASSED\n");
}

void test_tanh_backward()
{
    int shape[] = {1};
    Tensor *t = zf_tensor_create(shape, 1);
    assert(t);

    t->requires_grad = true;
    t->data[0] = 0.5f;

    Tensor *y = zf_tanh(t);
    assert(y);

    zf_backward(y);

    assert(t->grad != NULL);

    float expected = 1.0f - tanhf(0.5f) * tanhf(0.5f);
    assert(almost_equal(t->grad[0], expected, 1e-5f));

    zf_tensor_release(t);
    zf_tensor_release(y);
    printf("Test Tanh Backward: PASSED\n");
}

void test_leaky_relu_backward()
{
    int shape[] = {2};
    Tensor *x = zf_tensor_create(shape, 1);
    x->requires_grad = 1;
    x->data[0] = -2.0f;
    x->data[1] = 3.0f;

    Tensor *y = zf_leaky_relu(x, 0.1f);
    Tensor *loss = zf_tensor_sum_all(y);

    zf_backward(loss);

    assert(x->grad != NULL);
    assert(fabsf(x->grad[0] - 0.1f) < 1e-5f);
    assert(fabsf(x->grad[1] - 1.0f) < 1e-5f);

    zf_tensor_release(loss);
    zf_tensor_release(y);
    zf_tensor_release(x);
}

void test_sigmoid_backward()
{
    int shape[] = {1};
    Tensor *t = zf_tensor_create(shape, 1);
    assert(t);

    t->requires_grad = true;
    t->data[0] = 0.0f;

    Tensor *y = zf_sigmoid(t);
    assert(y);

    zf_backward(y);

    assert(t->grad != NULL);

    float s = 1.0f / (1.0f + expf(-t->data[0]));
    float expected = s * (1.0f - s);
    assert(almost_equal(t->grad[0], expected, 1e-6f));

    zf_tensor_release(t);
    zf_tensor_release(y);
    printf("Test Sigmoid Backward: PASSED\n");
}

int main()
{
    test_leaky_relu_forward();
    test_tanh_forward();
    test_softmax_1d_forward();

    test_tanh_autograd_graph();
    test_leaky_relu_autograd_graph();
    test_softmax_autograd_graph();

    test_tanh_backward();
    test_leaky_relu_backward();
    test_sigmoid_backward();

    printf("All activation tests PASSED\n");
    return 0;
}
