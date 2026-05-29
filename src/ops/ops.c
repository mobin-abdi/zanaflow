#include <zanaflow/ops/ops.h>
#include <zanaflow/autograd/autograd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int tensor_same_shape(const Tensor *a, const Tensor *b) {
    if (!a || !b) return 0;
    if (a->ndim != b->ndim) return 0;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    return 1;
}

typedef struct {
    int size;
} CtxSize;

typedef struct {
    int size;
} CtxSumAll;

typedef struct {
    int size;
    float inv_n;
} CtxMeanAll;

typedef struct {
    int rows;
    int cols;
} CtxAddBias;

static void ctx_free_default(void *p) { free(p); }

static void backward_add(AutogradNode *node, float *grad_out) {
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    if (!grad_out) { fprintf(stderr, "backward_add: grad_out is NULL\n"); return; }
    if (a->requires_grad) {
        tensor_ensure_grad(a);
        for (int i = 0; i < a->size; i++) a->grad[i] += grad_out[i]; // درسته
    }
    if (b->requires_grad) {
        tensor_ensure_grad(b);
        for (int i = 0; i < b->size; i++) b->grad[i] += grad_out[i]; // درسته
    }
}


static void backward_sub(AutogradNode *node, float *grad_out) {
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    if (!grad_out) { fprintf(stderr, "backward_sub: grad_out is NULL\n"); return; }

    if (a->requires_grad) {
        tensor_ensure_grad(a);
        for (int i = 0; i < a->size; i++) a->grad[i] += grad_out[i];
    }
    if (b->requires_grad) {
        tensor_ensure_grad(b);
        for (int i = 0; i < b->size; i++) b->grad[i] -= grad_out[i];
    }
}

static void backward_mul(AutogradNode *node, float *grad_out) {
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    if (!grad_out) { fprintf(stderr, "backward_mul: grad_out is NULL\n"); return; }
    if (a->requires_grad) {
        tensor_ensure_grad(a);
        for (int i = 0; i < a->size; i++) {
            a->grad[i] += grad_out[i] * b->data[i]; // این هم درسته!
        }
    }
    if (b->requires_grad) {
        tensor_ensure_grad(b);
        for (int i = 0; i < b->size; i++) {
            b->grad[i] += grad_out[i] * a->data[i]; // این هم درسته!
        }
    }
}

static void backward_sum_all(AutogradNode *node, float *grad_out) {
    Tensor *a = node->inputs[0];
    if (!grad_out) { fprintf(stderr, "backward_sum_all: grad_out is NULL\n"); return; }
    
    if (a->requires_grad) {
        tensor_ensure_grad(a);
        float go = grad_out[0];
        for (int i = 0; i < a->size; i++) {
            a->grad[i] += go;
        }
    }
}

static void backward_mean_all(AutogradNode *node, float *grad_out) {
    Tensor *a = node->inputs[0];
    if (!grad_out) { fprintf(stderr, "backward_mean_all: grad_out is NULL\n"); return; }
    
    if (a->requires_grad) {
        tensor_ensure_grad(a);
        float go = grad_out[0];
        float inv_n = 1.0f / (float)a->size;
        for (int i = 0; i < a->size; i++) {
            a->grad[i] += go * inv_n;
        }
    }
}

static void backward_add_bias(AutogradNode *node, float *grad_out) {
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    if (!grad_out) { fprintf(stderr, "backward_add_bias: grad_out is NULL\n"); return; }

    int rows = a->shape[0];
    int cols = a->shape[1];

    if (a->requires_grad) {
        tensor_ensure_grad(a);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                a->grad[r * cols + c] += grad_out[r * cols + c];
            }
        }
    }

    if (b->requires_grad) {
        tensor_ensure_grad(b);
        for (int j = 0; j < cols; j++) {
            float acc = 0.0f;
            for (int r = 0; r < rows; r++) {
                acc += grad_out[r * cols + j];
            }
            b->grad[j] += acc;
        }
    }
}

Tensor *tensor_add(Tensor *a, Tensor *b) {
    if (!tensor_same_shape(a, b)) return NULL;

    Tensor *out = tensor_create(a->shape, a->ndim);
    for (int i = 0; i < out->size; i++) out->data[i] = a->data[i] + b->data[i];

    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = 1;
        out->grad_node = autograd_node_create(backward_add, 2);

        out->grad_node->inputs[0] = a; tensor_retain(a);
        out->grad_node->inputs[1] = b; tensor_retain(b);

        out->grad_node->ctx = out;
        out->grad_node->ctx_free = NULL;
    }
    return out;
}

Tensor *tensor_sub(Tensor *a, Tensor *b) {
    if (!tensor_same_shape(a, b)) return NULL;

    Tensor *out = tensor_create(a->shape, a->ndim);
    for (int i = 0; i < out->size; i++) out->data[i] = a->data[i] - b->data[i];

    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = 1;
        out->grad_node = autograd_node_create(backward_sub, 2);

        out->grad_node->inputs[0] = a; tensor_retain(a);
        out->grad_node->inputs[1] = b; tensor_retain(b);

        out->grad_node->ctx = out;
        out->grad_node->ctx_free = NULL;
    }
    return out;
}

Tensor *tensor_mul(Tensor *a, Tensor *b) {
    if (!tensor_same_shape(a, b)) return NULL;

    Tensor *out = tensor_create(a->shape, a->ndim);
    for (int i = 0; i < out->size; i++) out->data[i] = a->data[i] * b->data[i];

    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = 1;
        out->grad_node = autograd_node_create(backward_mul, 2);

        out->grad_node->inputs[0] = a; tensor_retain(a);
        out->grad_node->inputs[1] = b; tensor_retain(b);

        out->grad_node->ctx = out;
        out->grad_node->ctx_free = NULL;
    }
    return out;
}

Tensor *tensor_sum_all(Tensor *a) {
    int s[1] = {1};
    Tensor *out = tensor_create(s, 1);
    float acc = 0.0f;
    for (int i = 0; i < a->size; i++) acc += a->data[i];
    out->data[0] = acc;

    if (a->requires_grad) {
        out->requires_grad = 1;
        out->grad_node = autograd_node_create(backward_sum_all, 1);

        out->grad_node->inputs[0] = a; tensor_retain(a);

        out->grad_node->ctx = out;
        out->grad_node->ctx_free = NULL;
    }
    return out;
}

Tensor *tensor_mean_all(Tensor *a) {
    int s[1] = {1};
    Tensor *out = tensor_create(s, 1);
    float acc = 0.0f;
    for (int i = 0; i < a->size; i++) acc += a->data[i];
    out->data[0] = acc / (float)a->size;

    if (a->requires_grad) {
        out->requires_grad = 1;
        out->grad_node = autograd_node_create(backward_mean_all, 1);

        out->grad_node->inputs[0] = a; tensor_retain(a);

        out->grad_node->ctx = out;
        out->grad_node->ctx_free = NULL;
    }
    return out;
}

// a: (rows, cols) , b: (cols)
Tensor *tensor_add_bias(Tensor *a, Tensor *b) {
    if (!a || !b) return NULL;
    if (a->ndim != 2) return NULL;
    if (b->ndim != 1) return NULL;
    int rows = a->shape[0];
    int cols = a->shape[1];
    if (b->shape[0] != cols) return NULL;

    Tensor *out = tensor_create(a->shape, a->ndim);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            out->data[idx] = a->data[idx] + b->data[c];
        }
    }

    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = 1;
        out->grad_node = autograd_node_create(backward_add_bias, 2);

        out->grad_node->inputs[0] = a; tensor_retain(a);
        out->grad_node->inputs[1] = b; tensor_retain(b);

        out->grad_node->ctx = out;
        out->grad_node->ctx_free = NULL;
    }
    return out;
}
