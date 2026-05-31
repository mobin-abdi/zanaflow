#include <stdlib.h>

#include <math.h>

#include <zanaflow/optim/adam.h>

Adam *zf_adam_create(Parameter **params, int count, float lr, float beta1, float beta2, float eps)
{

    if (!params || count <= 0)
    {
        return NULL;
    }

    Adam *opt = (Adam *)malloc(sizeof(Adam));
    if (!opt)
    {
        return NULL;
    }

    opt->params = (Parameter **)malloc(sizeof(Parameter *) * count);
    opt->m = (float **)malloc(sizeof(float *) * count);
    opt->v = (float **)malloc(sizeof(float *) * count);

    if (!opt->params || !opt->m || !opt->v)
    {
        free(opt->params);
        free(opt->m);
        free(opt->v);
        free(opt);
        return NULL;
    }

    opt->count = count;
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->t = 0;

    for (int i = 0; i < count; i++)
    {
        opt->params[i] = params[i];

        int sz = params[i]->value->size;
        opt->m[i] = (float *)calloc(sz, sizeof(float));
        opt->v[i] = (float *)calloc(sz, sizeof(float));

        if (!opt->m[i] || !opt->v[i])
        {
            for (int j = 0; j <= i; j++)
            {
                free(opt->m[j]);
                free(opt->v[j]);
            }
            free(opt->params);
            free(opt->m);
            free(opt->v);
            free(opt);
            return NULL;
        }
    }

    return opt;
}

void zf_adam_zero_grad(Adam *opt)
{

    if (!opt)
    {
        return;
    }

    for (int i = 0; i < opt->count; i++)
    {
        Parameter *p = opt->params[i];
        if (!p || !p->value || !p->value->grad)
        {
            continue;
        }

        for (int j = 0; j < p->value->size; j++)
        {
            p->value->grad[j] = 0.0f;
        }
    }
}

void zf_adam_step(Adam *opt)
{

    if (!opt)
    {
        return;
    }

    opt->t += 1;

    float b1_correction = 1.0f - powf(opt->beta1, (float)opt->t);
    float b2_correction = 1.0f - powf(opt->beta2, (float)opt->t);

    for (int i = 0; i < opt->count; i++)
    {
        Parameter *p = opt->params[i];
        if (!p || !p->value || !p->value->data || !p->value->grad)
        {
            continue;
        }

        Tensor *t = p->value;
        int sz = t->size;

        for (int j = 0; j < sz; j++)
        {
            float g = t->grad[j];

            opt->m[i][j] = opt->beta1 * opt->m[i][j] + (1.0f - opt->beta1) * g;
            opt->v[i][j] = opt->beta2 * opt->v[i][j] + (1.0f - opt->beta2) * g * g;

            float m_hat = opt->m[i][j] / b1_correction;
            float v_hat = opt->v[i][j] / b2_correction;

            t->data[j] -= opt->lr * m_hat / (sqrtf(v_hat) + opt->eps);
        }
    }
}

void zf_adam_free(Adam *opt)
{
    if (!opt)
    {
        return;
    }

    for (int i = 0; i < opt->count; i++)
    {
        free(opt->m[i]);
        free(opt->v[i]);
    }

    free(opt->params);
    free(opt->m);
    free(opt->v);
    free(opt);
}