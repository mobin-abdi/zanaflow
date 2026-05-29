#include <zanaflow/autograd/autograd.h>
#include <stdlib.h>
#include <stdio.h>

// --------------- autograd_node_create ---------------
AutogradNode *autograd_node_create(BackwardFn backward, int input_count) {
    AutogradNode *node = malloc(sizeof(AutogradNode));
    if (!node) return NULL;

    node->backward = backward;
    node->input_count = input_count;
    node->inputs = NULL;
    if (input_count > 0) {
        node->inputs = malloc(input_count * sizeof(Tensor*));
        if (!node->inputs) { free(node); return NULL; }
    }
    node->ctx = NULL;
    node->ctx_free = NULL;
    ref_init(&node->ref);
    node->visited = 0;
    return node;
}

// --------------- autograd_node_release ---------------
// !!! تعریف تابع autograd_node_release اضافه شد !!!
void autograd_node_release(AutogradNode *node) {
    if (!node) return;
    // اول هر حافظه‌ای که context داره رو آزاد کن
    if (node->ctx && node->ctx_free) {
        node->ctx_free(node->ctx);
    }
    // بعد ورودی‌ها رو آزاد کن (اینجا فقط اشاره‌گرها رو null می‌کنیم، چون retain/release خود تنسورها مدیریت می‌شه)
    if (node->inputs) {
        for (int i = 0; i < node->input_count; i++) {
            if (node->inputs[i]) {
                tensor_release(node->inputs[i]); // این مهم است!
            }
        }
        free(node->inputs);
    }
    // در نهایت خود نود رو آزاد کن
    free(node);
}


// --------------- build_topo ---------------
static void build_topo(AutogradNode *node, AutogradNode ***list, int *size, int *cap) {
    if (!node || node->visited) return;
    node->visited = 1;

    for (int i = 0; i < node->input_count; i++) {
        if (node->inputs[i] && node->inputs[i]->grad_node) {
            build_topo(node->inputs[i]->grad_node, list, size, cap);
        }
    }

    if (*size >= *cap) {
        *cap = (*cap == 0) ? 16 : (*cap * 2);
        *list = realloc(*list, (*cap) * sizeof(AutogradNode*));
        if (!*list) {
            fprintf(stderr, "Error reallocating topo list.\n");
            exit(1);
        }
    }
    (*list)[(*size)++] = node;
}


// --------------- zanaflow_backward ---------------
void zanaflow_backward(Tensor *loss) {
    if (!loss || !loss->grad_node) {
        fprintf(stderr, "Warning: Loss tensor has no grad_node.\n");
        return;
    }

    // === مهم: تنظیم گرادیان اولیه ===
    if (!tensor_ensure_grad(loss)) {
        fprintf(stderr, "Error: Could not allocate grad for loss.\n");
        return;
    }
    loss->grad[0] = 1.0f;   // ← این خط حیاتی بود!

    AutogradNode **topo = NULL;
    int size = 0, cap = 0;

    build_topo(loss->grad_node, &topo, &size, &cap);

    for (int i = size - 1; i >= 0; i--) {
        AutogradNode *node = topo[i];
        float *grad_output = NULL;

        if (node->ctx) {
            Tensor *output_tensor = (Tensor*)node->ctx;
            if (output_tensor && output_tensor->grad) {
                grad_output = output_tensor->grad;
            }
        }

        if (node->backward) {
            node->backward(node, grad_output);
        }

        node->visited = 0;
    }

    free(topo);
}
