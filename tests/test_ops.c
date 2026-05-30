#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <zanaflow/tensor/tensor.h>
#include <zanaflow/ops/ops.h>

int main()
{
    int shape[] = {2, 2};

    Tensor *a = zf_tensor_create(shape, 2);
    Tensor *b = zf_tensor_create(shape, 2);

    assert(a != NULL);
    assert(b != NULL);

    a->data[0] = 1.0f; a->data[1] = 2.0f;
    a->data[2] = 3.0f; a->data[3] = 4.0f;

    b->data[0] = 5.0f; b->data[1] = 6.0f;
    b->data[2] = 7.0f; b->data[3] = 8.0f;

    Tensor *c = zf_tensor_add(a, b);
    assert(c != NULL);
    assert(c->data[0] == 6.0f);
    assert(c->data[1] == 8.0f);
    assert(c->data[2] == 10.0f);
    assert(c->data[3] == 12.0f);

    Tensor *d = zf_tensor_sub(b, a);
    assert(d != NULL);
    assert(d->data[0] == 4.0f);
    assert(d->data[1] == 4.0f);
    assert(d->data[2] == 4.0f);
    assert(d->data[3] == 4.0f);

    Tensor *e = zf_tensor_mul(a, b);
    assert(e != NULL);
    assert(e->data[0] == 5.0f);
    assert(e->data[1] == 12.0f);
    assert(e->data[2] == 21.0f);
    assert(e->data[3] == 32.0f);

    Tensor *sum = zf_tensor_sum_all(a);
    assert(sum != NULL);
    assert(sum->ndim == 1 || sum->ndim == 0);
    assert(sum->size == 1);
    assert(fabsf(sum->data[0] - 10.0f) < 1e-6f);
    zf_tensor_release(sum);

    Tensor *mean = zf_tensor_mean_all(a);
    assert(mean != NULL);
    assert(mean->size == 1);
    assert(fabsf(mean->data[0] - 2.5f) < 1e-6f);
    zf_tensor_release(mean);

    zf_tensor_release(a);
    zf_tensor_release(b);
    zf_tensor_release(c);
    zf_tensor_release(d);
    zf_tensor_release(e);

    printf("All ops tests passed.\n");
    return 0;
}
