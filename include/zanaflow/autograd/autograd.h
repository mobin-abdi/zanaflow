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
    Tensor *output;
};

AutogradNode *zf_autograd_node_create(BackwardFn backward, int input_count);
void zf_autograd_node_release(AutogradNode *node);
void zf_backward(Tensor *loss);

#endif
