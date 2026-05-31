#include <zanaflow/core/parameter.h>
#include <stdlib.h>

Parameter *zf_parameter_create(int *shape, int ndim)
{
    Parameter *p = (Parameter *)malloc(sizeof(Parameter));
    if (!p)
    {
        return NULL;
    }

    p->value = zf_tensor_create(shape, ndim);
    if (!p->value)
    {
        free(p);
        return NULL;
    }

    p->value->requires_grad = 1;

    if (!zf_tensor_ensure_grad(p->value))
    {
        zf_tensor_release(p->value);
        free(p);
        return NULL;
    }

    zf_tensor_zero_grad(p->value);
    return p;
}

Parameter *zf_parameter_from_tensor(Tensor *t)
{
    if (!t)
    {
        return NULL;
    }

    Parameter *p = (Parameter *)malloc(sizeof(Parameter));
    if (!p)
    {
        return NULL;
    }

    p->value = t;
    zf_tensor_retain(t);

    p->value->requires_grad = 1;
    if (!zf_tensor_ensure_grad(p->value))
    {
        zf_tensor_release(p->value);
        free(p);
        return NULL;
    }

    return p;
}

void zf_parameter_free(Parameter *p)
{
    if (!p)
    {
        return;
    }

    if (p->value)
    {
        zf_tensor_release(p->value);
    }

    free(p);
}
