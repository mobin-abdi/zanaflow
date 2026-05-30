#include <zanaflow/autograd/autograd.h>
#include <stdlib.h>
#include <stdio.h>

AutogradNode *zf_autograd_node_create(BackwardFn backward, int input_count) 
{
    if (input_count < 0)
    {
        return NULL;
    }

    AutogradNode *node = malloc(sizeof(AutogradNode));
    if (!node)
    {
        return NULL;
    }

    node->backward = backward;
    node->input_count = input_count;
    node->inputs = NULL;
    node->ctx = NULL;
    node->ctx_free = NULL;
    node->visited = 0;
    ref_init(&node->ref);

    if (input_count > 0)
    {
        node->inputs = calloc((size_t)input_count, sizeof(Tensor *));
        if (!node->inputs)
        {
            free(node);
            return NULL;
        }
    }

    return node;
}

void zf_autograd_node_release(AutogradNode *node) 
{
    if (!node)
    {
        return;
    }

    if (!ref_release(&node->ref))
    {
        return;
    }

    if (node->ctx && node->ctx_free) 
    {
        node->ctx_free(node->ctx);
    }

    if (node->inputs) 
    {
        for (int i = 0; i < node->input_count; i++) 
        {
            if (node->inputs[i]) 
            {
                zf_tensor_release(node->inputs[i]);
            }
        }
        free(node->inputs);
    }

    free(node);
}

static void zf_build_topo(AutogradNode *node, AutogradNode ***list, int *size, int *cap) 
{
    if (!node || node->visited)
    {
        return;
    }
    node->visited = 1;

    for (int i = 0; i < node->input_count; i++) 
    {
        if (node->inputs[i] && node->inputs[i]->grad_node) 
        {
            zf_build_topo(node->inputs[i]->grad_node, list, size, cap);
        }
    }

    if (*size >= *cap) 
    {
        *cap = (*cap == 0) ? 16 : (*cap * 2);
        *list = realloc(*list, (*cap) * sizeof(AutogradNode*));
        if (!*list) 
        {
            fprintf(stderr, "Error reallocating topo list.\n");
            exit(1);
        }
    }
    (*list)[(*size)++] = node;
}

void zf_backward(Tensor *loss) 
{
    if (!loss || !loss->grad_node) 
    {
        fprintf(stderr, "Warning: Loss tensor has no grad_node.\n");
        return;
    }

    if (!zf_tensor_ensure_grad(loss)) 
    {
        fprintf(stderr, "Error: Could not allocate grad for loss.\n");
        return;
    }
    loss->grad[0] = 1.0f;

    AutogradNode **topo = NULL;
    int size = 0, cap = 0;

    zf_build_topo(loss->grad_node, &topo, &size, &cap);

    for (int i = size - 1; i >= 0; i--) 
    {
        AutogradNode *node = topo[i];
        float *grad_output = NULL;

        if (node->ctx) 
        {
            Tensor *output_tensor = (Tensor*)node->ctx;
            if (output_tensor && output_tensor->grad) 
            {
                grad_output = output_tensor->grad;
            }
        }

        if (node->backward) 
        {
            node->backward(node, grad_output);
        }

        node->visited = 0;
    }

    free(topo);
}
