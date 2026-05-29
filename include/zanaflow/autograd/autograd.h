#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <zanaflow/tensor/tensor.h>

typedef struct AutogradNode AutogradNode;

typedef void (*BackwardFn)(AutogradNode *node, float *grad_output);

struct AutogradNode {
    BackwardFn backward;
    Tensor **inputs;
    int input_count;
    void *ctx;
    void (*ctx_free)(void *);
    RefCount ref;
    int visited;
};

AutogradNode *autograd_node_create(BackwardFn backward, int input_count);
void autograd_node_release(AutogradNode *node);
void zanaflow_backward(Tensor *loss);

#endif
