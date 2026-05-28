#include <stdio.h>
#include <assert.h>
#include <zanaflow/tensor/tensor.h>

int main() {
    printf("--- Starting Tensor Tests ---\n");

    int shape[] = {2, 3};
    Tensor *t = tensor_create(shape, 2);
    
    assert(t != NULL);
    assert(t->ndim == 2);
    assert(t->size == 6);
    assert(t->shape[0] == 2);
    assert(t->shape[1] == 3);
    printf("Test 1: Creation - PASSED\n");

    tensor_fill(t, 7.5f);
    for(int i = 0; i < t->size; i++) {
        assert(t->data[i] == 7.5f);
    }
    printf("Test 2: Fill - PASSED\n");

    printf("Test 3: Visual Print Check:\n");
    tensor_print(t);

    tensor_free(t);
    printf("Test 4: Free Memory - PASSED (Check with Valgrind if possible)\n");

    printf("--- All Tests Passed! ---\n");
    return 0;
}
