// #include <stdio.h>
// #include <stdlib.h>
// #include <zanaflow.h>
// #include <zanaflow/autograd/autograd.h>
// // نیازی به parameter.h جداگانه نیست، چون Parameter را دستی می‌سازیم

// int main(void)
// {
//     int input_size = 1;
//     int output_size = 1;

//     int input_shape[] = {1, input_size};
//     Tensor *input_data = zf_tensor_create(input_shape, 2);
//     if (!input_data)
//     {
//         fprintf(stderr, "Error: Failed to create input_data tensor.\n");
//         return 1;
//     }
//     input_data->data[0] = 1.0f;

//     int target_shape[] = {1, output_size};
//     Tensor *target_data = zf_tensor_create(target_shape, 2);
//     if (!target_data)
//     {
//         fprintf(stderr, "Error: Failed to create target_data tensor.\n");
//         zf_tensor_release(input_data);
//         return 1;
//     }
//     target_data->data[0] = 3.0f;

//     int weight_shape[] = {input_size, output_size};
//     Tensor *weights = zf_tensor_create(weight_shape, 2);
//     if (!weights)
//     {
//         fprintf(stderr, "Error: Failed to create weights tensor.\n");
//         zf_tensor_release(input_data);
//         zf_tensor_release(target_data);
//         return 1;
//     }

//     int bias_shape[] = {1, output_size};
//     Tensor *bias = zf_tensor_create(bias_shape, 2);
//     if (!bias)
//     {
//         fprintf(stderr, "Error: Failed to create bias tensor.\n");
//         zf_tensor_release(input_data);
//         zf_tensor_release(target_data);
//         zf_tensor_release(weights);
//         return 1;
//     }

//     zf_init_he_uniform(weights, bias, input_size);

//     // تنظیم requires_grad و تخصیص grad برای تنسورها
//     weights->requires_grad = 1;
//     bias->requires_grad = 1;
//     zf_tensor_ensure_grad(weights); // grad را تخصیص می‌دهد (اگر NULL باشد)
//     zf_tensor_ensure_grad(bias);

//     // ساخت Parameterهای محلی (موقت، فقط برای دادن به بهینه‌ساز)
//     Parameter param_w, param_b;
//     param_w.value = weights;
//     param_b.value = bias;

//     Parameter *params[2] = {&param_w, &param_b};

//     float learning_rate = 0.01f;
//     SGD *sgd_optimizer = zf_sgd_create(params, 2, learning_rate);
//     if (!sgd_optimizer)
//     {
//         fprintf(stderr, "Error: Failed to create SGD optimizer.\n");
//         // نیازی به آزاد کردن param_w و param_b نیست، چون محلی هستند
//         zf_tensor_release(input_data);
//         zf_tensor_release(target_data);
//         zf_tensor_release(weights);
//         zf_tensor_release(bias);
//         return 1;
//     }

//     int epochs = 100;
//     for (int epoch = 0; epoch < epochs; ++epoch)
//     {
//         zf_sgd_zero_grad(sgd_optimizer);

//         Tensor *linear_output = zf_tensor_mat_mul(input_data, weights);
//         if (!linear_output)
//         {
//             fprintf(stderr, "Error during mat_mul.\n");
//             break;
//         }

//         Tensor *final_output = zf_tensor_add(linear_output, bias);
//         if (!final_output)
//         {
//             fprintf(stderr, "Error during add.\n");
//             zf_tensor_release(linear_output);
//             break;
//         }

//         Tensor *loss = zf_loss_mse(final_output, target_data);
//         if (!loss)
//         {
//             fprintf(stderr, "Error: loss is NULL.\n");
//             zf_tensor_release(final_output);
//             zf_tensor_release(linear_output);
//             break;
//         }

//         zf_backward(loss);
//         zf_sgd_step(sgd_optimizer);

//         if (epoch % 10 == 0)
//         {
//             float w_grad = weights->grad ? weights->grad[0] : -999.0f;
//             float b_grad = bias->grad ? bias->grad[0] : -999.0f;
//             printf("Epoch %d: loss=%.4f, W=%.4f, W_grad=%.4f, b=%.4f, b_grad=%.4f\n",
//                    epoch, loss->data[0], weights->data[0], w_grad, bias->data[0], b_grad);
//         }

//         zf_tensor_release(loss);
//         zf_tensor_release(final_output);
//         zf_tensor_release(linear_output);
//     }

//     zf_sgd_free(sgd_optimizer);
//     zf_tensor_release(input_data);
//     zf_tensor_release(target_data);
//     zf_tensor_release(weights);
//     zf_tensor_release(bias);

//     printf("Training finished!\n");
//     return 0;
// }

// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>

// #include <zanaflow.h>
// #include <zanaflow/autograd/autograd.h>

// // ---- utilities ----
// static float frand_uniform(float a, float b)
// {
//     float r = (float)rand() / (float)RAND_MAX;
//     return a + (b - a) * r;
// }

// static void fill_dataset_sin(Tensor *X, Tensor *Y)
// {
//     // X: [N,1], Y: [N,1]
//     int N = X->shape[0];
//     for (int i = 0; i < N; i++)
//     {
//         float x = frand_uniform(-3.1415926f, 3.1415926f); // [-pi, pi]
//         X->data[i] = x;
//         Y->data[i] = sinf(x);
//     }
// }

// static float mse_on_dataset(Tensor *pred, Tensor *target)
// {
//     // pred/target: [N,1]
//     float acc = 0.0f;
//     for (int i = 0; i < pred->size; i++)
//     {
//         float d = pred->data[i] - target->data[i];
//         acc += d * d;
//     }
//     return acc / (float)pred->size;
// }

// int main(void)
// {
//     srand(0);

//     // ---- dataset ----
//     const int N = 128;
//     int x_shape[] = {N, 1};
//     int y_shape[] = {N, 1};

//     Tensor *X = zf_tensor_create(x_shape, 2);
//     Tensor *Y = zf_tensor_create(y_shape, 2);
//     if (!X || !Y)
//     {
//         fprintf(stderr, "Failed to create dataset tensors.\n");
//         return 1;
//     }

//     fill_dataset_sin(X, Y);

//     // ---- model: 1 -> 16 -> 1 ----
//     const int in_dim = 1;
//     const int hidden = 16;
//     const int out_dim = 1;

//     int W1_shape[] = {in_dim, hidden};  // [1,16]
//     int b1_shape[] = {hidden};          // [16]
//     int W2_shape[] = {hidden, out_dim}; // [16,1]
//     int b2_shape[] = {out_dim};         // [1]

//     Tensor *W1 = zf_tensor_create(W1_shape, 2);
//     Tensor *b1 = zf_tensor_create(b1_shape, 1);
//     Tensor *W2 = zf_tensor_create(W2_shape, 2);
//     Tensor *b2 = zf_tensor_create(b2_shape, 1);

//     if (!W1 || !b1 || !W2 || !b2)
//     {
//         fprintf(stderr, "Failed to create model parameters.\n");
//         return 1;
//     }

//     // init
//     zf_init_he_uniform(W1, b1, in_dim);
//     zf_init_he_uniform(W2, b2, hidden);

//     W1->requires_grad = 1;
//     b1->requires_grad = 1;
//     W2->requires_grad = 1;
//     b2->requires_grad = 1;

//     Parameter params[4];
//     params[0].value = W1;
//     params[1].value = b1;
//     params[2].value = W2;
//     params[3].value = b2;

//     float lr = 0.05f;
//     SGD *opt = zf_sgd_create(params, 4, lr);
//     if (!opt)
//     {
//         fprintf(stderr, "Failed to create optimizer.\n");
//         return 1;
//     }

//     // ---- training ----
//     const int epochs = 2000;

//     for (int epoch = 0; epoch < epochs; epoch++)
//     {
//         zf_sgd_zero_grad(opt);

//         // forward
//         Tensor *z1 = zf_tensor_mat_mul(X, W1);   // [N,16]
//         Tensor *a1 = zf_tensor_add_bias(z1, b1); // [N,16]
//         Tensor *h = zf_tanh(a1);                 // [N,16]

//         Tensor *z2 = zf_tensor_mat_mul(h, W2);     // [N,1]
//         Tensor *pred = zf_tensor_add_bias(z2, b2); // [N,1]

//         Tensor *loss = zf_loss_mse(pred, Y); // scalar

//         if (!z1 || !a1 || !h || !z2 || !pred || !loss)
//         {
//             fprintf(stderr, "Forward produced NULL at epoch %d\n", epoch);
//             break;
//         }

//         // backward
//         zf_backward(loss);

//         // capture grads and params BEFORE step
//         float gW1 = (W1->grad) ? W1->grad[0] : 0.0f;
//         float gW2 = (W2->grad) ? W2->grad[0] : 0.0f;
//         float gb1 = (b1->grad) ? b1->grad[0] : 0.0f;
//         float gb2 = (b2->grad) ? b2->grad[0] : 0.0f;

//         float w1v = W1->data[0];
//         float w2v = W2->data[0];
//         float b1v = b1->data[0];
//         float b2v = b2->data[0];

//         if (epoch % 200 == 0)
//         {
//             float train_mse = mse_on_dataset(pred, Y);

//             printf("Epoch %4d | loss=%.6f | mse=%.6f | W1[0]=%.4f g=%.6f | W2[0]=%.4f g=%.6f\n",
//                    epoch, loss->data[0], train_mse, w1v, gW1, w2v, gW2);

//             printf("... | b1[0]=%.4f g=%.6f | b2[0]=%.4f g=%.6f\n",
//                    b1v, gb1, b2v, gb2);
//         }

//         // step
//         zf_sgd_step(opt);

//         // free forward tensors
//         zf_tensor_release(loss);
//         zf_tensor_release(pred);
//         zf_tensor_release(z2);
//         zf_tensor_release(h);
//         zf_tensor_release(a1);
//         zf_tensor_release(z1);
//     }

//     // ---- cleanup ----
//     zf_sgd_free(opt);

//     zf_tensor_release(X);
//     zf_tensor_release(Y);

//     zf_tensor_release(W1);
//     zf_tensor_release(b1);
//     zf_tensor_release(W2);
//     zf_tensor_release(b2);

//     printf("Done.\n");
//     return 0;
// }

// #include <stdio.h>
// #include <zanaflow.h>
// #include <zanaflow/autograd/autograd.h>

// static void print_array(const char *name, float *x, int n)
// {
//     printf("%s = [", name);
//     for (int i = 0; i < n; i++)
//     {
//         printf("%.4f", x[i]);
//         if (i + 1 < n)
//             printf(", ");
//     }
//     printf("]\n");
// }

// int main(void)
// {
//     int a_shape[] = {2, 3};
//     int b_shape[] = {3};

//     Tensor *a = zf_tensor_create(a_shape, 2);
//     Tensor *b = zf_tensor_create(b_shape, 1);

//     if (!a || !b)
//     {
//         fprintf(stderr, "Failed to create tensors.\n");
//         return 1;
//     }

//     a->requires_grad = 1;
//     b->requires_grad = 1;

//     // a = [[1,2,3],
//     //      [4,5,6]]
//     a->data[0] = 1.0f;
//     a->data[1] = 2.0f;
//     a->data[2] = 3.0f;
//     a->data[3] = 4.0f;
//     a->data[4] = 5.0f;
//     a->data[5] = 6.0f;

//     // b = [10,20,30]
//     b->data[0] = 10.0f;
//     b->data[1] = 20.0f;
//     b->data[2] = 30.0f;

//     Tensor *y = zf_tensor_add_bias(a, b); // [2,3]
//     Tensor *loss = zf_tensor_sum_all(y);  // scalar

//     if (!y || !loss)
//     {
//         fprintf(stderr, "Forward failed.\n");
//         zf_tensor_release(a);
//         zf_tensor_release(b);
//         return 1;
//     }

//     printf("Forward:\n");
//     print_array("a.data", a->data, a->size);
//     print_array("b.data", b->data, b->size);
//     print_array("y.data", y->data, y->size);
//     printf("loss = %.4f\n", loss->data[0]);

//     zf_backward(loss);

//     printf("\nAfter backward:\n");
//     if (a->grad)
//         print_array("a.grad", a->grad, a->size);
//     else
//         printf("a.grad = NULL\n");

//     if (b->grad)
//         print_array("b.grad", b->grad, b->size);
//     else
//         printf("b.grad = NULL\n");

//     printf("\nExpected:\n");
//     printf("a.grad = [1,1,1,1,1,1]\n");
//     printf("b.grad = [2,2,2]\n");

//     zf_tensor_release(loss);
//     zf_tensor_release(y);
//     zf_tensor_release(a);
//     zf_tensor_release(b);

//     return 0;
// }

// #include <stdio.h>
// #include <math.h>
// #include <zanaflow.h>
// #include <zanaflow/autograd/autograd.h>

// static int nearly_equal(float a, float b, float eps)
// {
//     return fabsf(a - b) < eps;
// }

// static void print_arr(const char *name, float *x, int n)
// {
//     printf("%s = [", name);
//     for (int i = 0; i < n; i++)
//     {
//         printf("%.4f", x[i]);
//         if (i + 1 < n)
//             printf(", ");
//     }
//     printf("]\n");
// }

// int main(void)
// {
//     int shape[] = {2, 3}; // no broadcasting assumptions

//     Tensor *x = zf_tensor_create(shape, 2);
//     Tensor *w1 = zf_tensor_create(shape, 2);
//     Tensor *w2 = zf_tensor_create(shape, 2);

//     if (!x || !w1 || !w2)
//     {
//         printf("alloc failed\n");
//         return 1;
//     }

//     x->requires_grad = 1;
//     w1->requires_grad = 1;
//     w2->requires_grad = 1;

//     // Fill with deterministic values
//     // x = [1..6]
//     for (int i = 0; i < 6; i++)
//         x->data[i] = (float)(i + 1);

//     // w1 = all 2, w2 = all 3  => w1+w2 = 5
//     for (int i = 0; i < 6; i++)
//     {
//         w1->data[i] = 2.0f;
//         w2->data[i] = 3.0f;
//     }

//     Tensor *y1 = zf_tensor_mul(x, w1);
//     Tensor *y2 = zf_tensor_mul(x, w2);
//     Tensor *z = zf_tensor_add(y1, y2);
//     Tensor *L = zf_tensor_sum_all(z); // scalar

//     if (!y1 || !y2 || !z || !L)
//     {
//         printf("forward failed\n");
//         return 1;
//     }

//     zf_backward(L);

//     printf("x.data:\n");
//     print_arr("x", x->data, 6);
//     printf("w1.data:\n");
//     print_arr("w1", w1->data, 6);
//     printf("w2.data:\n");
//     print_arr("w2", w2->data, 6);

//     printf("\nGrads:\n");
//     if (x->grad)
//         print_arr("x.grad", x->grad, 6);
//     else
//         printf("x.grad NULL\n");
//     if (w1->grad)
//         print_arr("w1.grad", w1->grad, 6);
//     else
//         printf("w1.grad NULL\n");
//     if (w2->grad)
//         print_arr("w2.grad", w2->grad, 6);
//     else
//         printf("w2.grad NULL\n");

//     // Expected:
//     // dL/dx = w1 + w2 = 5 everywhere
//     // dL/dw1 = x, dL/dw2 = x
//     int ok = 1;
//     for (int i = 0; i < 6; i++)
//     {
//         float ex_dx = 5.0f;
//         float ex_dw1 = x->data[i];
//         float ex_dw2 = x->data[i];

//         if (!x->grad || !w1->grad || !w2->grad)
//         {
//             ok = 0;
//             break;
//         }

//         if (!nearly_equal(x->grad[i], ex_dx, 1e-5f))
//             ok = 0;
//         if (!nearly_equal(w1->grad[i], ex_dw1, 1e-5f))
//             ok = 0;
//         if (!nearly_equal(w2->grad[i], ex_dw2, 1e-5f))
//             ok = 0;
//     }

//     printf("\nExpected:\n");
//     printf("x.grad  = [5,5,5,5,5,5]\n");
//     printf("w1.grad = x.data\n");
//     printf("w2.grad = x.data\n");

//     printf("\nRESULT: %s\n", ok ? "PASS" : "FAIL");

//     // cleanup
//     zf_tensor_release(L);
//     zf_tensor_release(z);
//     zf_tensor_release(y2);
//     zf_tensor_release(y1);
//     zf_tensor_release(x);
//     zf_tensor_release(w1);
//     zf_tensor_release(w2);

//     return ok ? 0 : 2;
// }
