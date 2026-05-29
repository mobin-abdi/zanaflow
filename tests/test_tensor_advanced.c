#include <stdio.h>
#include <assert.h>
#include <zanaflow/tensor/tensor.h>
#include <zanaflow/ops/ops.h>

void test_clone() {
    int shape[] = {2, 3};
    Tensor *t = tensor_create(shape, 2);
    tensor_fill(t, 5.5f);
    
    Tensor *cloned = tensor_clone(t);
    
    assert(cloned != NULL);
    assert(cloned->size == t->size);
    assert(cloned->ndim == t->ndim);
    assert(cloned->data != t->data); // باید آدرس‌ها متفاوت باشند
    for(int i=0; i<t->size; i++) assert(cloned->data[i] == 5.5f);
    
    tensor_free(t);
    tensor_free(cloned);
    printf("Test Clone: PASSED\n");
}

void test_zeros_ones() {
    int shape[] = {10};
    Tensor *z = tensor_zeros(shape, 1);
    Tensor *o = tensor_ones(shape, 1);
    
    for(int i=0; i<10; i++) {
        assert(z->data[i] == 0.0f);
        assert(o->data[i] == 1.0f);
    }
    
    tensor_free(z);
    tensor_free(o);
    printf("Test Zeros/Ones: PASSED\n");
}

void test_reshape() {
    int shape[] = {2, 3}; // size = 6
    Tensor *t = tensor_create(shape, 2);
    
    int new_shape[] = {6, 1};
    Tensor *reshaped = tensor_reshape(t, new_shape, 2);
    assert(reshaped != NULL);
    assert(reshaped->shape[0] == 6);
    assert(reshaped->size == 6);

    int bad_shape[] = {5, 5};
    Tensor *bad = tensor_reshape(t, bad_shape, 2);
    assert(bad == NULL); // باید نال برگرداند چون سایز همخوانی ندارد
    
    tensor_free(t);
    tensor_free(reshaped);
    printf("Test Reshape: PASSED\n");
}

void test_scalar_ops() {
    int shape[] = {2, 2};
    Tensor *t = tensor_ones(shape, 2); // همه ۱
    
    Tensor *added = tensor_add_scalar(t, 2.5f); // همه ۳.۵
    Tensor *mul = tensor_mul_scalar(t, 10.0f); // همه ۱۰
    
    assert(added->data[0] == 3.5f);
    assert(mul->data[0] == 10.0f);
    
    tensor_free(t);
    tensor_free(added);
    tensor_free(mul);
    printf("Test Scalar Ops: PASSED\n");
}

int main() {
    printf("--- Running Advanced Tensor Tests ---\n");
    test_clone();
    test_zeros_ones();
    test_reshape();
    test_scalar_ops();
    printf("--- All Tests Passed! ---\n");
    return 0;
}
