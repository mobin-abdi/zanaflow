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
    node->output = NULL;
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
        *list = realloc(*list, (*cap) * sizeof(AutogradNode *));
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
    if (!loss)
    {
        fprintf(stderr, "zf_backward: loss is NULL\n");
        return;
    }

    if (!loss->requires_grad)
        return;

    if (loss->size != 1)
    {
        fprintf(stderr, "zf_backward: expected scalar loss (size==1), got size=%d. "
                        "Call sum_all/mean_all first.\n",
                loss->size);
        return;
    }

    if (!loss->grad_node)
    {
        if (!zf_tensor_ensure_grad(loss))
        {
            return;
        }
        loss->grad[0] = 1.0f;
        return;
    }

    if (!zf_tensor_ensure_grad(loss))
    {
        fprintf(stderr, "zf_backward: could not allocate grad for loss\n");
        return;
    }
    loss->grad[0] = 1.0f;

    AutogradNode **topo = NULL;
    int n = 0, cap = 0;

    zf_build_topo(loss->grad_node, &topo, &n, &cap);

    for (int i = n - 1; i >= 0; i--)
    {
        AutogradNode *node = topo[i];
        if (!node)
        {
            continue;
        }

        if (!node->backward)
        {
            node->visited = 0;
            continue;
        }

        if (!node->output)
        {
            fprintf(stderr, "zf_backward: node->output is NULL (op must set node->output)\n");
            node->visited = 0;
            continue;
        }

        if (!zf_tensor_ensure_grad(node->output))
        {
            fprintf(stderr, "zf_backward: could not allocate grad for node output\n");
            node->visited = 0;
            continue;
        }

        node->backward(node, node->output->grad);
        node->visited = 0;
    }

    free(topo);
}
