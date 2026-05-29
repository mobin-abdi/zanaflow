// #include <stdio.h>
// #include <assert.h>
// #include <math.h>
// #include <zanaflow/tensor/tensor.h>
// #include <zanaflow/ops/activations.h>

// void test_relu() {
//     int shape[] = {4};
//     Tensor *t = tensor_create(shape, 1);
    
//     // مقداردهی با مقادیر مثبت و منفی
//     t->data[0] = -2.0f;
//     t->data[1] = 0.0f;
//     t->data[2] = 3.5f;
//     t->data[3] = -0.5f;

//     Tensor *activated = relu(t);
    
//     assert(activated != NULL);
//     assert(activated->data[0] == 0.0f);
//     assert(activated->data[1] == 0.0f);
//     assert(activated->data[2] == 3.5f);
//     assert(activated->data[3] == 0.0f);

//     tensor_free(t);
//     tensor_free(activated);
//     printf("Test ReLU: PASSED\n");
// }

// void test_sigmoid() {
//     int shape[] = {3};
//     Tensor *t = tensor_create(shape, 1);
    
//     t->data[0] = 0.0f;   // sigmoid(0) = 0.5
//     t->data[1] = 2.0f;   // sigmoid(2) ~= 0.880797
//     t->data[2] = -2.0f;  // sigmoid(-2) ~= 0.119202

//     Tensor *activated = sigmoid(t);
    
//     assert(activated != NULL);
    
//     // تست با تولرانس خطای محاسبات اعشاری (EPSILON)
//     float eps = 1e-5f;
//     assert(fabsf(activated->data[0] - 0.5f) < eps);
//     assert(fabsf(activated->data[1] - 0.880797f) < eps);
//     assert(fabsf(activated->data[2] - 0.119202f) < eps);

//     tensor_free(t);
//     tensor_free(activated);
//     printf("Test Sigmoid: PASSED\n");
// }

// int main() {
//     printf("--- Running Activation Tests ---\n");
//     test_relu();
//     test_sigmoid();
//     printf("--- All Activation Tests Passed! ---\n");
//     return 0;
// }

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <zanaflow/tensor/tensor.h>
#include <zanaflow/ops/activations.h>

static int almost_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

void test_leaky_relu()
{
    int shape[] = {4};
    Tensor *t = tensor_create(shape, 1);
    t->data[0] = -2.0f;
    t->data[1] = 0.0f;
    t->data[2] = 3.0f;
    t->data[3] = -1.0f;

    Tensor *y = leaky_relu_activation(t, 0.1f);
    assert(y);

    assert(almost_equal(y->data[0], -0.2f, 1e-6f));
    assert(almost_equal(y->data[1],  0.0f, 1e-6f));
    assert(almost_equal(y->data[2],  3.0f, 1e-6f));
    assert(almost_equal(y->data[3], -0.1f, 1e-6f));

    tensor_free(t);
    tensor_free(y);
    printf("Test LeakyReLU: PASSED\n");
}

void test_tanh()
{
    int shape[] = {3};
    Tensor *t = tensor_create(shape, 1);
    t->data[0] = 0.0f;
    t->data[1] = 1.0f;
    t->data[2] = -1.0f;

    Tensor *y = tanh_activation(t);
    assert(y);

    assert(almost_equal(y->data[0], 0.0f, 1e-6f));
    assert(almost_equal(y->data[1], tanhf(1.0f), 1e-6f));
    assert(almost_equal(y->data[2], tanhf(-1.0f), 1e-6f));

    tensor_free(t);
    tensor_free(y);
    printf("Test Tanh: PASSED\n");
}

void test_softmax_1d()
{
    int shape[] = {3};
    Tensor *t = tensor_create(shape, 1);

    // ورودی ساده
    t->data[0] = 1.0f;
    t->data[1] = 2.0f;
    t->data[2] = 3.0f;

    Tensor *y = softmax_activation(t);
    assert(y);

    float sum = y->data[0] + y->data[1] + y->data[2];
    assert(almost_equal(sum, 1.0f, 1e-5f));

    // ترتیب باید حفظ بشه: بزرگ‌ترین ورودی بزرگ‌ترین خروجی
    assert(y->data[2] > y->data[1]);
    assert(y->data[1] > y->data[0]);

    tensor_free(t);
    tensor_free(y);
    printf("Test Softmax 1D: PASSED\n");
}

int main()
{
    test_leaky_relu();
    test_tanh();
    test_softmax_1d();
    printf("All activation tests PASSED\n");
    return 0;
}
