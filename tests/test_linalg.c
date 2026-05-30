#include <stdio.h>
#include <assert.h>

#include <zanaflow/tensor/tensor.h>
#include <zanaflow/ops/linalg.h>

int main(void)
{
    int a_shape[2] = {2, 3};
    int b_shape[2] = {3, 2};

    Tensor *A = zf_tensor_create(a_shape, 2);
    Tensor *B = zf_tensor_create(b_shape, 2);
    assert(A && B);

    A->data[0]=1; A->data[1]=2; A->data[2]=3;
    A->data[3]=4; A->data[4]=5; A->data[5]=6;

    B->data[0]=7;  B->data[1]=8;
    B->data[2]=9;  B->data[3]=10;
    B->data[4]=11; B->data[5]=12;

    Tensor *C = zf_tensor_matmul_2d(A, B);
    assert(C);
    assert(C->ndim == 2);
    assert(C->shape[0] == 2 && C->shape[1] == 2);

    assert(C->data[0] == 58.0f);
    assert(C->data[1] == 64.0f);
    assert(C->data[2] == 139.0f);
    assert(C->data[3] == 154.0f);

    zf_tensor_release(A);
    zf_tensor_release(B);
    zf_tensor_release(C);

    printf("Test MatMul 2D: PASSED\n");
    return 0;
}
