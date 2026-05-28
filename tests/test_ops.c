#include <assert.h>
#include <stdio.h>
#include <zanaflow/tensor/tensor.h>
#include <zanaflow/ops/ops.h>

int main()
{
    int shape[] = {2, 2};

    Tensor *a = tensor_create(shape, 2);
    Tensor *b = tensor_create(shape, 2);

    assert(a != NULL);
    assert(b != NULL);

    a->data[0] = 1.0f; a->data[1] = 2.0f;
    a->data[2] = 3.0f; a->data[3] = 4.0f;

    b->data[0] = 5.0f; b->data[1] = 6.0f;
    b->data[2] = 7.0f; b->data[3] = 8.0f;

    Tensor *c = tensor_add(a, b);
    assert(c != NULL);
    assert(c->data[0] == 6.0f);
    assert(c->data[1] == 8.0f);
    assert(c->data[2] == 10.0f);
    assert(c->data[3] == 12.0f);

    Tensor *d = tensor_sub(b, a);
    assert(d != NULL);
    assert(d->data[0] == 4.0f);
    assert(d->data[1] == 4.0f);
    assert(d->data[2] == 4.0f);
    assert(d->data[3] == 4.0f);

    Tensor *e = tensor_mul(a, b);
    assert(e != NULL);
    assert(e->data[0] == 5.0f);
    assert(e->data[1] == 12.0f);
    assert(e->data[2] == 21.0f);
    assert(e->data[3] == 32.0f);

    float sum = tensor_sum_all(a);
    assert(sum == 10.0f);

    float mean = tensor_mean_all(a);
    assert(mean == 2.5f);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);
    tensor_free(e);

    printf("All ops tests passed.\n");
    return 0;
}
