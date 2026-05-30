#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <zanaflow/nn/dense.h>

#define EPS 1e-5f

static int nearly_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

static void fail(const char *msg) {
    fprintf(stderr, "TEST FAILED: %s\n", msg);
    exit(1);
}

int main() {
    DenseLayer *layer = zf_dense_create(2, 3);
    if (!layer) fail("zf_dense_create returned NULL");

    int in_shape[] = {1, 2};
    Tensor *input = zf_tensor_create(in_shape, 2);
    if (!input) fail("zf_tensor_create(input) returned NULL");


    input->data[0] = 0.5f;
    input->data[1] = -0.2f;

    layer->weights->data[0] = 1.0fک
    layer->weights->data[1] = 2.0f;
    layer->weights->data[2] = 3.0f;
    layer->weights->data[3] = 4.0f;
    layer->weights->data[4] = 5.0f;
    layer->weights->data[5] = 6.0f;

    layer->bias->data[0] = 0.1f;
    layer->bias->data[1] = 0.2f;
    layer->bias->data[2] = 0.3f;

    Tensor *output = zf_dense_forward(layer, input);
    if (!output) fail("zf_dense_forward returned NULL");
 
    if (output->ndim != 2) fail("output ndim should be 2");
    if (output->shape[0] != 1 || output->shape[1] != 3)
        fail("output shape should be [1, 3]");


    float expected[3] = {-0.2f, 0.2f, 0.6f};

    for (int i = 0; i < 3; i++) {
        if (!nearly_equal(output->data[i], expected[i], EPS)) {
            fprintf(stderr,
                    "TEST FAILED: output[%d] = %f, expected = %f\n",
                    i, output->data[i], expected[i]);
            zf_tensor_print(output);
            zf_tensor_release(input);
            zf_tensor_release(output);
            zf_dense_free(layer);
            return 1;
        }
    }

    printf("TEST PASSED: dense forward output is correct.\n");

    zf_tensor_release(input);
    zf_tensor_release(output);
    zf_dense_free(layer);
    return 0;
}
