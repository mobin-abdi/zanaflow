#include <zanaflow/optim/sgd.h>
#include <stdlib.h>

static void apply_sgd_update(Tensor *value, float lr)
{
    if (!value || !value->data || !value->grad)
    {
        return;
    }

    for (int i = 0; i < value->size; i++)
    {
        value->data[i] -= lr * value->grad[i];
    }
}

SGD *zf_sgd_create(Parameter **params, int count, float lr)
{
    if (!params || count <= 0)
    {
        return NULL;
    }

    SGD *opt = malloc(sizeof(SGD));
    if (!opt)
    {
        return NULL;
    }

    opt->params = params;
    opt->count = count;
    opt->lr = lr;
    return opt;
}

void zf_sgd_zero_grad(SGD *opt)
{
    if (!opt)
    {
        return;
    }

    for (int i = 0; i < opt->count; i++)
    {
        Parameter *p = opt->params[i];
        Tensor *value = p ? p->value : NULL;
        if (value)
        {
            zf_tensor_zero_grad(value);
        }
    }
}

void zf_sgd_step(SGD *opt)
{
    if (!opt)
    {
        return;
    }

    for (int i = 0; i < opt->count; i++)
    {
        Parameter *p = opt->params[i];
        Tensor *value = p ? p->value : NULL;
        if (value && value->grad)
        {
            apply_sgd_update(value, opt->lr);
        }
    }
}

void zf_sgd_free(SGD *opt)
{
    if (!opt)
    {
        return;
    }

    free(opt);
}
