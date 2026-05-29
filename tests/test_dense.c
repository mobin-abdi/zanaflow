#include <stdio.h>
#include <zanaflow/nn/dense.h>

int main() {
    DenseLayer *layer = dense(2, 3);

    int in_shape[] = {1, 2};
    Tensor *input = tensor_create(in_shape, 2);
    input->data[0] = 0.5f;
    input->data[1] = -0.2f;

    Tensor *output = dense_forward(layer, input);

    printf("Weights Matrix:\n");
    tensor_print(layer->weights);
    printf("\nInput:\n");
    tensor_print(input);
    printf("\nOutput (Forward Pass):\n");
    tensor_print(output);

    tensor_free(input);
    tensor_free(output);
    dense_free(layer);

    return 0;
}
