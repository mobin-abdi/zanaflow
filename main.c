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
//     if (!input_data) {
//         fprintf(stderr, "Error: Failed to create input_data tensor.\n");
//         return 1;
//     }
//     input_data->data[0] = 1.0f;

//     int target_shape[] = {1, output_size};
//     Tensor *target_data = zf_tensor_create(target_shape, 2);
//     if (!target_data) {
//         fprintf(stderr, "Error: Failed to create target_data tensor.\n");
//         zf_tensor_release(input_data);
//         return 1;
//     }
//     target_data->data[0] = 3.0f;

//     int weight_shape[] = {input_size, output_size};
//     Tensor *weights = zf_tensor_create(weight_shape, 2);
//     if (!weights) {
//         fprintf(stderr, "Error: Failed to create weights tensor.\n");
//         zf_tensor_release(input_data);
//         zf_tensor_release(target_data);
//         return 1;
//     }

//     int bias_shape[] = {1, output_size};
//     Tensor *bias = zf_tensor_create(bias_shape, 2);
//     if (!bias) {
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
//     zf_tensor_ensure_grad(weights);   // grad را تخصیص می‌دهد (اگر NULL باشد)
//     zf_tensor_ensure_grad(bias);

//     // ساخت Parameterهای محلی (موقت، فقط برای دادن به بهینه‌ساز)
//     Parameter param_w, param_b;
//     param_w.value = weights;
//     param_b.value = bias;

//     Parameter *params[2] = { &param_w, &param_b };

//     float learning_rate = 0.01f;
//     SGD *sgd_optimizer = zf_sgd_create(params, 2, learning_rate);
//     if (!sgd_optimizer) {
//         fprintf(stderr, "Error: Failed to create SGD optimizer.\n");
//         // نیازی به آزاد کردن param_w و param_b نیست، چون محلی هستند
//         zf_tensor_release(input_data);
//         zf_tensor_release(target_data);
//         zf_tensor_release(weights);
//         zf_tensor_release(bias);
//         return 1;
//     }

//     int epochs = 100;
//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         zf_sgd_zero_grad(sgd_optimizer);

//         Tensor *linear_output = zf_tensor_mat_mul(input_data, weights);
//         if (!linear_output) {
//             fprintf(stderr, "Error during mat_mul.\n");
//             break;
//         }

//         Tensor *final_output = zf_tensor_add(linear_output, bias);
//         if (!final_output) {
//             fprintf(stderr, "Error during add.\n");
//             zf_tensor_release(linear_output);
//             break;
//         }

//         Tensor *loss = zf_loss_mse(final_output, target_data);
//         if (!loss) {
//             fprintf(stderr, "Error: loss is NULL.\n");
//             zf_tensor_release(final_output);
//             zf_tensor_release(linear_output);
//             break;
//         }

//         zf_backward(loss);
//         zf_sgd_step(sgd_optimizer);

//         if (epoch % 10 == 0) {
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
//         if (i + 1 < n) printf(", ");
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
//     a->data[0] = 1.0f; a->data[1] = 2.0f; a->data[2] = 3.0f;
//     a->data[3] = 4.0f; a->data[4] = 5.0f; a->data[5] = 6.0f;

//     // b = [10,20,30]
//     b->data[0] = 10.0f;
//     b->data[1] = 20.0f;
//     b->data[2] = 30.0f;

//     Tensor *y = zf_tensor_add_bias(a, b);   // [2,3]
//     Tensor *loss = zf_tensor_sum_all(y);    // scalar

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
//     if (a->grad) print_array("a.grad", a->grad, a->size);
//     else printf("a.grad = NULL\n");

//     if (b->grad) print_array("b.grad", b->grad, b->size);
//     else printf("b.grad = NULL\n");

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

// static int nearly_equal(float a, float b, float eps) {
//     return fabsf(a - b) < eps;
// }

// static void print_arr(const char *name, float *x, int n) {
//     printf("%s = [", name);
//     for (int i = 0; i < n; i++) {
//         printf("%.4f", x[i]);
//         if (i + 1 < n) printf(", ");
//     }
//     printf("]\n");
// }

// int main(void)
// {
//     int shape[] = {2, 3}; // no broadcasting assumptions

//     Tensor *x  = zf_tensor_create(shape, 2);
//     Tensor *w1 = zf_tensor_create(shape, 2);
//     Tensor *w2 = zf_tensor_create(shape, 2);

//     if (!x || !w1 || !w2) { printf("alloc failed\n"); return 1; }

//     x->requires_grad  = 1;
//     w1->requires_grad = 1;
//     w2->requires_grad = 1;

//     // Fill with deterministic values
//     // x = [1..6]
//     for (int i = 0; i < 6; i++) x->data[i] = (float)(i + 1);

//     // w1 = all 2, w2 = all 3  => w1+w2 = 5
//     for (int i = 0; i < 6; i++) { w1->data[i] = 2.0f; w2->data[i] = 3.0f; }

//     Tensor *y1 = zf_tensor_mul(x, w1);
//     Tensor *y2 = zf_tensor_mul(x, w2);
//     Tensor *z  = zf_tensor_add(y1, y2);
//     Tensor *L  = zf_tensor_sum_all(z); // scalar

//     if (!y1 || !y2 || !z || !L) { printf("forward failed\n"); return 1; }

//     zf_backward(L);

//     printf("x.data:\n");  print_arr("x", x->data, 6);
//     printf("w1.data:\n"); print_arr("w1", w1->data, 6);
//     printf("w2.data:\n"); print_arr("w2", w2->data, 6);

//     printf("\nGrads:\n");
//     if (x->grad)  print_arr("x.grad",  x->grad,  6); else printf("x.grad NULL\n");
//     if (w1->grad) print_arr("w1.grad", w1->grad, 6); else printf("w1.grad NULL\n");
//     if (w2->grad) print_arr("w2.grad", w2->grad, 6); else printf("w2.grad NULL\n");

//     // Expected:
//     // dL/dx = w1 + w2 = 5 everywhere
//     // dL/dw1 = x, dL/dw2 = x
//     int ok = 1;
//     for (int i = 0; i < 6; i++) {
//         float ex_dx  = 5.0f;
//         float ex_dw1 = x->data[i];
//         float ex_dw2 = x->data[i];

//         if (!x->grad || !w1->grad || !w2->grad) { ok = 0; break; }

//         if (!nearly_equal(x->grad[i], ex_dx, 1e-5f))  ok = 0;
//         if (!nearly_equal(w1->grad[i], ex_dw1, 1e-5f)) ok = 0;
//         if (!nearly_equal(w2->grad[i], ex_dw2, 1e-5f)) ok = 0;
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

// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>

// #include <zanaflow.h>
// #include <zanaflow/autograd/autograd.h>

// static int nearly_equal(float a, float b, float eps) {
//     float diff = fabsf(a - b);
//     float scale = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
//     return diff <= eps * scale;
// }

// static void print_arr(const char *name, float *x, int n) {
//     printf("%s = [", name);
//     for (int i = 0; i < n; i++) {
//         printf("%.6f", x[i]);
//         if (i + 1 < n) printf(", ");
//     }
//     printf("]\n");
// }

// static int check_arr_eq(const char *name, float *got, float *exp, int n, float eps) {
//     for (int i = 0; i < n; i++) {
//         if (!nearly_equal(got[i], exp[i], eps)) {
//             printf("FAIL %s[%d]: got=%.8f exp=%.8f\n", name, i, got[i], exp[i]);
//             return 0;
//         }
//     }
//     return 1;
// }

/* =========================
   TEST 1: Shared parameter reuse
   y1 = x*w
   y2 = x*w   (reuse same w)
   z = y1+y2
   L = sum(z)
   Expected:
     dL/dx = 2*w
     dL/dw = 2*x
   ========================= */
// static int test_shared_param_reuse(void) {
//     printf("\n=== Test 1: Shared Param Reuse ===\n");

//     int shape[] = {2, 3};
//     const int n = 6;

//     Tensor *x = zf_tensor_create(shape, 2);
//     Tensor *w = zf_tensor_create(shape, 2);
//     x->requires_grad = 1;
//     w->requires_grad = 1;

//     for (int i = 0; i < n; i++) x->data[i] = (float)(i + 1); // 1..6
//     for (int i = 0; i < n; i++) w->data[i] = 2.0f;

//     Tensor *y1 = zf_tensor_mul(x, w);
//     Tensor *y2 = zf_tensor_mul(x, w); // reuse SAME w pointer
//     Tensor *z  = zf_tensor_add(y1, y2);
//     Tensor *L  = zf_tensor_sum_all(z);

//     zf_backward(L);

//     float exp_dx[6];
//     float exp_dw[6];
//     for (int i = 0; i < n; i++) {
//         exp_dx[i] = 2.0f * w->data[i]; // 4
//         exp_dw[i] = 2.0f * x->data[i]; // 2,4,6,8,10,12
//     }

//     print_arr("x.grad", x->grad, n);
//     print_arr("w.grad", w->grad, n);

//     int ok = 1;
//     ok &= check_arr_eq("x.grad", x->grad, exp_dx, n, 1e-5f);
//     ok &= check_arr_eq("w.grad", w->grad, exp_dw, n, 1e-5f);

//     printf("RESULT: %s\n", ok ? "PASS" : "FAIL");

//     zf_tensor_release(L);
//     zf_tensor_release(z);
//     zf_tensor_release(y2);
//     zf_tensor_release(y1);
//     zf_tensor_release(x);
//     zf_tensor_release(w);

//     return ok;
// }

/* =========================
   TEST 2: Diamond graph on intermediate
   u = x*w
   z = u + u   (reuse SAME intermediate)
   L = sum(z)
   Expected:
     dL/du = 2*ones
     so dL/dx = 2*w
        dL/dw = 2*x
   This catches bugs where a node's grad is overwritten
   instead of accumulated when reused.
   ========================= */
// static int test_diamond_graph_intermediate(void) {
//     printf("\n=== Test 2: Diamond Graph (Intermediate Reuse) ===\n");

//     int shape[] = {2, 3};
//     const int n = 6;

//     Tensor *x = zf_tensor_create(shape, 2);
//     Tensor *w = zf_tensor_create(shape, 2);
//     x->requires_grad = 1;
//     w->requires_grad = 1;

//     for (int i = 0; i < n; i++) x->data[i] = (float)(i + 1); // 1..6
//     for (int i = 0; i < n; i++) w->data[i] = 3.0f;

//     Tensor *u = zf_tensor_mul(x, w);
//     Tensor *z = zf_tensor_add(u, u);      // reuse SAME u pointer
//     Tensor *L = zf_tensor_sum_all(z);

//     zf_backward(L);

//     float exp_dx[6];
//     float exp_dw[6];
//     for (int i = 0; i < n; i++) {
//         exp_dx[i] = 2.0f * w->data[i]; // 6
//         exp_dw[i] = 2.0f * x->data[i];
//     }

//     print_arr("x.grad", x->grad, n);
//     print_arr("w.grad", w->grad, n);

//     int ok = 1;
//     ok &= check_arr_eq("x.grad", x->grad, exp_dx, n, 1e-5f);
//     ok &= check_arr_eq("w.grad", w->grad, exp_dw, n, 1e-5f);

//     printf("RESULT: %s\n", ok ? "PASS" : "FAIL");

//     zf_tensor_release(L);
//     zf_tensor_release(z);
//     zf_tensor_release(u);
//     zf_tensor_release(x);
//     zf_tensor_release(w);

//     return ok;
// }

/* =========================
   Helper: forward loss for grad-check.
   We'll use: L = sum_all( (x*w) * (x*w) )
   i.e. L = sum( u^2 ), where u = x*w
   This is always scalar and smooth.
   ========================= */
// static float forward_loss_value(const float *x_data, const float *w_data, int n) {
//     float L = 0.0f;
//     for (int i = 0; i < n; i++) {
//         float u = x_data[i] * w_data[i];
//         L += u * u;
//     }
//     return L;
// }

/* =========================
   TEST 3: Numerical gradient check (finite differences)
   Compare autograd gradients with numeric gradients for the same graph:
     u = x*w
     v = u*u (elementwise)
     L = sum(v)
   Expected analytic:
     dL/dx = 2*(x*w)*w
     dL/dw = 2*(x*w)*x
   But we don't rely on formula; we compare autograd to finite-diff.
   ========================= */
// static int test_numerical_grad_check(void) {
//     printf("\n=== Test 3: Numerical Gradient Check ===\n");

//     int shape[] = {2, 3};
//     const int n = 6;
//     const float eps = 1e-3f;     // finite diff step
//     const float tol = 2e-2f;     // tolerance (float32 + nonlinearity + eps)

//     Tensor *x = zf_tensor_create(shape, 2);
//     Tensor *w = zf_tensor_create(shape, 2);
//     x->requires_grad = 1;
//     w->requires_grad = 1;

//     // pick non-trivial values (avoid zeros)
//     float x0[6] = {0.5f, -1.2f, 2.0f, -0.7f, 1.5f, -2.5f};
//     float w0[6] = {1.1f, -0.3f, 0.8f, 2.2f, -1.4f, 0.6f};

//     for (int i = 0; i < n; i++) { x->data[i] = x0[i]; w->data[i] = w0[i]; }

//     // autograd forward
//     Tensor *u = zf_tensor_mul(x, w);
//     Tensor *v = zf_tensor_mul(u, u);
//     Tensor *L = zf_tensor_sum_all(v);

//     zf_backward(L);

//     // numeric grads
//     float num_dx[6], num_dw[6];

//     // Make local copies to perturb
//     float x_tmp[6], w_tmp[6];
//     for (int i = 0; i < n; i++) { x_tmp[i] = x0[i]; w_tmp[i] = w0[i]; }

//     // dL/dx_i
//     for (int i = 0; i < n; i++) {
//         float old = x_tmp[i];

//         x_tmp[i] = old + eps;
//         float Lp = forward_loss_value(x_tmp, w_tmp, n);

//         x_tmp[i] = old - eps;
//         float Lm = forward_loss_value(x_tmp, w_tmp, n);

//         x_tmp[i] = old;
//         num_dx[i] = (Lp - Lm) / (2.0f * eps);
//     }

//     // dL/dw_i
//     for (int i = 0; i < n; i++) {
//         float old = w_tmp[i];

//         w_tmp[i] = old + eps;
//         float Lp = forward_loss_value(x_tmp, w_tmp, n);

//         w_tmp[i] = old - eps;
//         float Lm = forward_loss_value(x_tmp, w_tmp, n);

//         w_tmp[i] = old;
//         num_dw[i] = (Lp - Lm) / (2.0f * eps);
//     }

//     print_arr("x.grad (autograd)", x->grad, n);
//     print_arr("x.grad (numeric )", num_dx, n);
//     print_arr("w.grad (autograd)", w->grad, n);
//     print_arr("w.grad (numeric )", num_dw, n);

//     int ok = 1;
//     for (int i = 0; i < n; i++) {
//         if (!nearly_equal(x->grad[i], num_dx[i], tol)) {
//             printf("FAIL dL/dx[%d]: autograd=%.8f numeric=%.8f\n", i, x->grad[i], num_dx[i]);
//             ok = 0;
//         }
//         if (!nearly_equal(w->grad[i], num_dw[i], tol)) {
//             printf("FAIL dL/dw[%d]: autograd=%.8f numeric=%.8f\n", i, w->grad[i], num_dw[i]);
//             ok = 0;
//         }
//     }

//     printf("RESULT: %s (tol=%g, eps=%g)\n", ok ? "PASS" : "FAIL", tol, eps);

//     zf_tensor_release(L);
//     zf_tensor_release(v);
//     zf_tensor_release(u);
//     zf_tensor_release(x);
//     zf_tensor_release(w);

//     return ok;
// }

// int main(void) {
//     int ok = 1;

//     ok &= test_shared_param_reuse();
//     if (!ok) return 2;

//     ok &= test_diamond_graph_intermediate();
//     if (!ok) return 3;

//     ok &= test_numerical_grad_check();
//     if (!ok) return 4;

//     printf("\nALL 3 TESTS: PASS\n");
//     return 0;
// }

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <zanaflow.h>
#include <zanaflow/autograd/autograd.h>
#include <zanaflow/nn/dense.h>
#include <zanaflow/optim/adam.h>

static float frand_uniform(float a, float b)
{
    return a + (b - a) * (float)rand() / (float)RAND_MAX;
}

static void fill_dataset_sin(Tensor *X, Tensor *Y)
{
    if (!X || !Y)
        return;
    if (X->ndim != 2 || Y->ndim != 2)
        return;
    if (X->shape[0] != Y->shape[0])
        return;
    if (X->shape[1] != 1 || Y->shape[1] != 1)
        return;

    int n = X->shape[0];
    for (int i = 0; i < n; i++)
    {
        float x = frand_uniform(-3.1415926f, 3.1415926f);
        X->data[i] = x;
        Y->data[i] = sinf(x);
    }
}

static float compute_mse_raw(Tensor *pred, Tensor *target)
{
    if (!pred || !target)
        return 0.0f;
    if (pred->size != target->size)
        return 0.0f;
    float acc = 0.0f;
    for (int i = 0; i < pred->size; i++)
    {
        float d = pred->data[i] - target->data[i];
        acc += d * d;
    }
    return acc / (float)pred->size;
}

static void print_param_stats(const char *name, Parameter *p)
{
    if (!p || !p->value)
    {
        printf("%s: <null>\n", name);
        return;
    }
    Tensor *t = p->value;
    float v0 = (t->data && t->size > 0) ? t->data[0] : 0.0f;
    float g0 = (t->grad && t->size > 0) ? t->grad[0] : 0.0f;
    printf("%s: req=%d size=%d v0=%.6f g0=%.6f\n", name, t->requires_grad, t->size, v0, g0);
}

int main(void)
{
    srand(0);

    const int N = 128;
    int x_shape[] = {N, 1};
    int y_shape[] = {N, 1};

    Tensor *X = zf_tensor_create(x_shape, 2);
    Tensor *Y = zf_tensor_create(y_shape, 2);
    if (!X || !Y)
    {
        fprintf(stderr, "failed to create dataset tensors\n");
        return 1;
    }
    X->requires_grad = 0;
    Y->requires_grad = 0;
    fill_dataset_sin(X, Y);

    DenseLayer *fc1 = zf_dense_create(1, 16);
    DenseLayer *fc2 = zf_dense_create(16, 1);

    if (!fc1 || !fc2)
    {
        fprintf(stderr, "failed to create dense layers\n");
        zf_tensor_release(X);
        zf_tensor_release(Y);
        if (fc1)
            zf_dense_free(fc1);
        if (fc2)
            zf_dense_free(fc2);
        return 1;
    }

    printf("Initial parameter flags:\n");
    print_param_stats("fc1.W", fc1->weights);
    print_param_stats("fc1.b", fc1->bias);
    print_param_stats("fc2.W", fc2->weights);
    print_param_stats("fc2.b", fc2->bias);

    zf_init_he_uniform(fc1->weights->value, fc1->bias->value, fc1->in_features);
    zf_init_he_uniform(fc2->weights->value, fc2->bias->value, fc2->in_features);

    Parameter *params[4] = {0};
    if (zf_dense_parameters(fc1, &params[0]) != 2 ||
        zf_dense_parameters(fc2, &params[2]) != 2)
    {
        fprintf(stderr, "failed to collect parameters\n");
        zf_dense_free(fc1);
        zf_dense_free(fc2);
        zf_tensor_release(X);
        zf_tensor_release(Y);
        return 1;
    }

    Adam *opt = zf_adam_create(params, 4, 0.001f, 0.9f, 0.999f, 1e-2f);
    if (!opt)
    {
        fprintf(stderr, "failed to create optimizer\n");
        zf_dense_free(fc1);
        zf_dense_free(fc2);
        zf_tensor_release(X);
        zf_tensor_release(Y);
        return 1;
    }

    const int epochs = 2000;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        zf_adam_zero_grad(opt);

        Tensor *h1 = zf_dense_forward(fc1, X);
        if (!h1)
        {
            fprintf(stderr, "h1 forward failed at epoch %d\n", epoch);
            break;
        }

        Tensor *a1 = zf_tanh(h1);
        if (!a1)
        {
            fprintf(stderr, "tanh failed at epoch %d\n", epoch);
            zf_tensor_release(h1);
            break;
        }

        Tensor *pred = zf_dense_forward(fc2, a1);
        if (!pred)
        {
            fprintf(stderr, "pred forward failed at epoch %d\n", epoch);
            zf_tensor_release(a1);
            zf_tensor_release(h1);
            break;
        }

        Tensor *loss = zf_loss_mse(pred, Y);
        if (!loss)
        {
            fprintf(stderr, "loss failed at epoch %d\n", epoch);
            zf_tensor_release(pred);
            zf_tensor_release(a1);
            zf_tensor_release(h1);
            break;
        }

        if (epoch == 0)
        {
            printf("Sanity: pred->requires_grad=%d loss->requires_grad=%d\n", pred->requires_grad, loss->requires_grad);
        }

        zf_backward(loss);

        if (epoch % 200 == 0)
        {
            float mse = compute_mse_raw(pred, Y);
            printf("Epoch %4d | loss=%.6f | mse=%.6f\n", epoch, (loss->data ? loss->data[0] : 0.0f), mse);
            print_param_stats("fc1.W", fc1->weights);
            print_param_stats("fc1.b", fc1->bias);
            print_param_stats("fc2.W", fc2->weights);
            print_param_stats("fc2.b", fc2->bias);
        }

        zf_adam_step(opt);

        zf_tensor_release(loss);
        zf_tensor_release(pred);
        zf_tensor_release(a1);
        zf_tensor_release(h1);
    }

    zf_adam_free(opt);
    zf_dense_free(fc1);
    zf_dense_free(fc2);
    zf_tensor_release(X);
    zf_tensor_release(Y);

    printf("Training finished.\n");
    return 0;
}
