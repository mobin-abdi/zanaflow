#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <zanaflow/optim/rmsprop.h>
#include <zanaflow/tensor/tensor.h>

static int zf_param_numel(const Parameter *p)
{
    if (!p || !p->value)
    {
        return 0;
    }
    return p->value->size;
}

static void zf_rmsprop_free_partial(RMSProp *opt)
{
    if (!opt)
    {
        return;
    }

    if (opt->cache)
    {
        for (int i = 0; i < opt->count; i++)
        {
            free(opt->cache[i]);
        }
        free(opt->cache);
    }

    free(opt->params);
    free(opt);
}

RMSProp *zf_rmsprop_create(Parameter **params, int count, float lr, float alpha, float eps)
{
    if (!params)
    {
        fprintf(stderr, "zf_rmsprop_create: params cannot be NULL\n");
        return NULL;
    }

    if (count <= 0)
    {
        fprintf(stderr, "zf_rmsprop_create: count must be > 0\n");
        return NULL;
    }

    if (lr <= 0.0f)
    {
        fprintf(stderr, "zf_rmsprop_create: lr must be > 0\n");
        return NULL;
    }

    if (alpha < 0.0f || alpha >= 1.0f)
    {
        fprintf(stderr, "zf_rmsprop_create: alpha must be in [0, 1)\n");
        return NULL;
    }

    if (eps <= 0.0f)
    {
        fprintf(stderr, "zf_rmsprop_create: eps must be > 0\n");
        return NULL;
    }

    RMSProp *opt = (RMSProp *)calloc(1, sizeof(RMSProp));
    if (!opt)
    {
        fprintf(stderr, "zf_rmsprop_create: failed to allocate RMSProp\n");
        return NULL;
    }

    opt->count = count;
    opt->lr = lr;
    opt->alpha = alpha;
    opt->eps = eps;

    opt->params = (Parameter **)calloc((size_t)count, sizeof(Parameter *));
    opt->cache = (float **)calloc((size_t)count, sizeof(float *));
    if (!opt->params || !opt->cache)
    {
        fprintf(stderr, "zf_rmsprop_create: failed to allocate optimizer buffers\n");
        zf_rmsprop_free_partial(opt);
        return NULL;
    }

    for (int i = 0; i < count; i++)
    {
        if (!params[i] || !params[i]->value)
        {
            fprintf(stderr, "zf_rmsprop_create: params[%d] or params[%d]->value is NULL\n", i, i);
            zf_rmsprop_free_partial(opt);
            return NULL;
        }

        int n = zf_param_numel(params[i]);
        if (n <= 0)
        {
            fprintf(stderr, "zf_rmsprop_create: params[%d] has invalid size\n", i);
            zf_rmsprop_free_partial(opt);
            return NULL;
        }

        opt->params[i] = params[i];
        opt->cache[i] = (float *)calloc((size_t)n, sizeof(float));
        if (!opt->cache[i])
        {
            fprintf(stderr, "zf_rmsprop_create: failed to allocate cache for params[%d]\n", i);
            zf_rmsprop_free_partial(opt);
            return NULL;
        }
    }

    return opt;
}

int zf_rmsprop_zero_grad(RMSProp *opt)
{
    if (!opt)
    {
        fprintf(stderr, "zf_rmsprop_zero_grad: opt cannot be NULL\n");
        return -1;
    }

    for (int i = 0; i < opt->count; i++)
    {
        Parameter *p = opt->params[i];
        if (!p || !p->value)
        {
            fprintf(stderr, "zf_rmsprop_zero_grad: invalid parameter at index %d\n", i);
            return -1;
        }

        zf_tensor_zero_grad(p->value);
    }

    return 0;
}

int zf_rmsprop_step(RMSProp *opt)
{
    if (!opt)
    {
        fprintf(stderr, "zf_rmsprop_step: opt cannot be NULL\n");
        return -1;
    }

    for (int i = 0; i < opt->count; i++)
    {
        Parameter *p = opt->params[i];
        if (!p || !p->value)
        {
            fprintf(stderr, "zf_rmsprop_step: invalid parameter at index %d\n", i);
            return -1;
        }

        Tensor *t = p->value;

        if (!t->requires_grad)
        {
            continue;
        }

        if (!t->grad)
        {
            continue;
        }

        if (!opt->cache[i])
        {
            fprintf(stderr, "zf_rmsprop_step: cache[%d] is NULL\n", i);
            return -1;
        }

        for (int j = 0; j < t->size; j++)
        {
            float g = t->grad[j];
            opt->cache[i][j] = opt->alpha * opt->cache[i][j] + (1.0f - opt->alpha) * g * g;

            t->data[j] -= opt->lr * g / (sqrtf(opt->cache[i][j]) + opt->eps);
        }
    }

    return 0;
}

void zf_rmsprop_free(RMSProp *opt)
{
    if (!opt)
    {
        return;
    }

    if (opt->cache)
    {
        for (int i = 0; i < opt->count; i++)
        {
            free(opt->cache[i]);
        }
        free(opt->cache);
    }

    free(opt->params);
    free(opt);
}
