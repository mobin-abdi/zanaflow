#include <zanaflow/optim/sgd.h>
#include <stdlib.h>
#include <string.h>

static void zero_grad_tensor(Tensor *grad) {
    if (!grad || !grad->data) return;
    tensor_fill(grad, 0.0f);
}

static void apply_sgd_update(Tensor *value, const Tensor *grad, float lr) {
    if (!value || !grad || !value->data || !grad->data || value->size != grad->size) {
        return; 
    }

    for (int i = 0; i < value->size; i++) {
        value->data[i] -= lr * grad->data[i];
    }
}

SGD *sgd_create(Parameter *params, int count, float lr) {
    if (!params || count <= 0) {
        return NULL;
    }

    SGD *opt = (SGD *)malloc(sizeof(SGD));
    if (!opt) {
        return NULL;
    }

    opt->params = params;
    opt->count = count;
    opt->lr = lr;
    return opt;
}

void sgd_zero_grad(SGD *opt) {
    if (!opt) return;

    for (int i = 0; i < opt->count; i++) {
        if (opt->params[i].grad) {
            zero_grad_tensor(opt->params[i].grad);
        }
    }
}

void sgd_step(SGD *opt) {
    if (!opt) return;

    for (int i = 0; i < opt->count; i++) {
        if (opt->params[i].value && opt->params[i].grad) {
            apply_sgd_update(opt->params[i].value, opt->params[i].grad, opt->lr);
        }
    }
}

void sgd_free(SGD *opt) {
    if (!opt) return;
    free(opt);
}
