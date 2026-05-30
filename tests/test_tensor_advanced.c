#include <stdio.h>
#include <assert.h>
#include <zanaflow/tensor/tensor.h>

void test_clone()
{
    int shape[] = {2, 3};
    Tensor *t = zf_tensor_create(shape, 2);
    assert(t != NULL);

    zf_tensor_fill(t, 5.5f);

    Tensor *cloned = zf_tensor_clone(t);
    assert(cloned != NULL);
    assert(cloned->size == t->size);
    assert(cloned->ndim == t->ndim);
    assert(cloned->data != t->data);

    for (int i = 0; i < t->size; i++)
    {
        assert(cloned->data[i] == 5.5f);
    }

    zf_tensor_release(t);
    zf_tensor_release(cloned);

    printf("Test Clone: PASSED\n");
}

int main()
{
    printf("--- Running Tensor Tests ---\n");
    test_clone();
    printf("--- All Tests Passed! ---\n");
    return 0;
}
