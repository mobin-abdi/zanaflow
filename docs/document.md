# Zanaflow Tutorial – Learning the Sine Function with a Neural Network  
## آموزش زانافلو – یادگیری تابع سینوس با یک شبکه عصبی

درود!  
اگه به تازگی با **Zanaflow** آشنا شدی و می‌خوای ببینی چطور میشه باهاش یه شبکه عصبی واقعی تربیت کرد، جای درستی اومدی. توی این آموزش قدم‌به‌قدم یه شبکه ساده رو با کتابخونهٔ C خودمون می‌سازیم که تابع *sin(x)* رو یاد بگیره. کل کد آخر همین صفحه‌ست، می‌تونی کپی کنی و اجراش کنی.

Hey there!  
If you just got to know **Zanaflow** and want to see how to train a real neural network with it, you’re in the right place. In this tutorial we’ll build a simple network step by step using our C library to learn the *sin(x)* function. The whole code is at the bottom – you can copy, paste, and run it.

---

## What is Zanaflow? / زانافلو چیه؟
Zanaflow یه کتابخونهٔ یادگیری عمیق سبُک به زبان C هست که موتور مشتق‌گیری خودکار (autograd) داره. یعنی می‌تونی بدون اینکه خودت مشتق حساب کنی، مدل‌های عصبی رو آموزش بدی. کلیهٔ عملیات تنسوری مثل *mat_mul*، *add*، توابع فعال‌سازی (tanh, relu) و لایه‌های *Dense* توی خودش هست. تازه کلی هم بهینه‌ساز مثل *SGD* و *Adam و RMSprop* داره.

Zanaflow is a lightweight deep learning library written in C with an automatic differentiation engine (autograd). That means you can train neural networks without calculating derivatives manually. All tensor operations like *mat_mul*, *add*, activation functions (tanh, relu), and layers like `Dense` are built in. Plus it comes with optimizers like *SGD* and *Adam RMSprop*.

---

## The Task: Approximate sin(x) / کاری که می‌خوایم بکنیم
ما می‌خوایم یه شبکه عصبی کوچیک تربیت کنیم که برای ورودی `x` (یه عدد بین `-π` تا `π`) خروجی‌ای نزدیک به *sin(x)* بده. داده‌ها رو خودمون تصادفی می‌سازیم و بعد مدل رو آموزش می‌دیم.

We want to train a tiny neural network that takes a number `x` (between `-π` and `π`) and outputs something close to *sin(x)*. We’ll generate the data ourselves randomly and then train the model.

---

## Step 1: Imports & Helper Functions / قدم اول: کتابخونه‌ها و توابع کمکی

اول از همه سرآیندهای مورد نیاز رو اضافه می‌کنیم.  
First, we include the necessary headers.

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <zanaflow.h>
#include <zanaflow/autograd/autograd.h>
#include <zanaflow/nn/dense.h>
#include <zanaflow/optim/adam.h>
```

بعدش یه تابع ساده برای ساخت داده‌های تصادفی مینویسیم.  
Then we write a simple function to generate random data.

```c
static float frand_uniform(float a, float b) {
    return a + (b - a) * (float)rand() / (float)RAND_MAX;
}

static void fill_dataset_sin(Tensor *X, Tensor *Y) {
    int n = X->shape[0];
    for (int i = 0; i < n; i++) {
        float x = frand_uniform(-3.1415926f, 3.1415926f);
        X->data[i] = x;
        Y->data[i] = sinf(x);
    }
}
```

همین‌طور یه تابع برای محاسبهٔ دستی MSE (برای نمایش توی ترمینال) و یکی برای چاپ وضعیت پارامترها اضافه می‌کنیم.  
We also add a function to compute MSE manually (just for logging) and one to print parameter stats.

```c
static float compute_mse_raw(Tensor *pred, Tensor *target) {
    float acc = 0.0f;
    for (int i = 0; i < pred->size; i++) {
        float d = pred->data[i] - target->data[i];
        acc += d * d;
    }
    return acc / (float)pred->size;
}

static void print_param_stats(const char *name, Parameter *p) {
    if (!p || !p->value) return;
    Tensor *t = p->value;
    float v0 = (t->data && t->size > 0) ? t->data[0] : 0.0f;
    float g0 = (t->grad && t->size > 0) ? t->grad[0] : 0.0f;
    printf("%s: req=%d size=%d v0=%.6f g0=%.6f\n",
           name, t->requires_grad, t->size, v0, g0);
}
```

---

## Step 2: Create the Data Tensors / قدم دوم: ساخت تنسورهای داده

حالا داخل `main` تنسورهای ورودی (`X`) و هدف (`Y`) رو می‌سازیم. ۱۲۸ نقطه تصادفی کافیه.  
Now inside `main` we create input (`X`) and target (`Y`) tensors. 128 random points will be enough.

```c
srand(0);
const int N = 128;
int x_shape[] = {N, 1};
int y_shape[] = {N, 1};

Tensor *X = zf_tensor_create(x_shape, 2);
Tensor *Y = zf_tensor_create(y_shape, 2);
X->requires_grad = 0;
Y->requires_grad = 0;
fill_dataset_sin(X, Y);
```

---

## Step 3: Build the Model / قدم سوم: ساختن مدل

مدل ما دو لایهٔ `Dense` داره:  
- لایهٔ اول: ۱ ورودی -> ۱۶ نورون پنهان  
- تابع فعال‌سازی `tanh`  
- لایهٔ دوم: ۱۶ -> ۱ خروجی  

Our model has two `Dense` layers:  
- First layer: 1 input -> 16 hidden neurons  
- `tanh` activation  
- Second layer: 16 -> 1 output  

```c
DenseLayer *fc1 = zf_dense_create(1, 16);
DenseLayer *fc2 = zf_dense_create(16, 1);
```

بعد وزن‌ها رو با روش `He` مقداردهی می‌کنیم (برای tanh و ReLU عالیه).  
Then we initialize the weights with the `He` method (great for tanh & ReLU).

```c
zf_init_he_uniform(fc1->weights->value, fc1->bias->value, fc1->in_features);
zf_init_he_uniform(fc2->weights->value, fc2->bias->value, fc2->in_features);
```

---

## Step 4: Collect Parameters & Create Optimizer / قدم چهارم: جمع‌آوری پارامترها و ساختن بهینه‌ساز

باید به بهینه‌ساز بگیم کدوم پارامترها رو به‌روز کنه. با `zf_dense_parameters` پارامترهای هر لایه رو می‌گیریم و توی یه آرایه می‌ذاریم.  
We need to tell the optimizer which parameters to update. `zf_dense_parameters` gives us the parameters of each layer, and we put them in an array.

```c
Parameter *params[4] = {0};
zf_dense_parameters(fc1, &params[0]);
zf_dense_parameters(fc2, &params[2]);
```

حالا بهینه‌ساز `Adam` رو با نرخ یادگیری ۰.۰۰۱ و تنظیمات استاندارد می‌سازیم.  
Now we create the `Adam` optimizer with learning rate 0.001 and standard betas.

```c
Adam *opt = zf_adam_create(params, 4, 0.001f, 0.9f, 0.999f, 1e-2f);
```

---

## Step 5: Training Loop / قدم پنجم: حلقهٔ آموزش

قلب برنامه اینجاست. برای هر اپوک:  

1. **صفر کردن گرادیان‌ها** (zero_grad)  
2. **پاس رو به جلو**:  
   - `h1 = fc1(X)`  
   - `a1 = tanh(h1)`  
   - `pred = fc2(a1)`  
   - `loss = MSE(pred, Y)`  
3. **انتشار رو به عقب** (`zf_backward(loss)`)  
4. **یک گام بهینه‌ساز** (`step`)  
5. **آزادسازی حافظه** (تنسورهای میانی)  

The heart of the program. For each epoch:  

1. **Zero the gradients** (zero_grad)  
2. **Forward pass**:  
   - `h1 = fc1(X)`  
   - `a1 = tanh(h1)`  
   - `pred = fc2(a1)`  
   - `loss = MSE(pred, Y)`  
3. **Backward pass** (`zf_backward(loss)`)  
4. **Optimizer step** (`step`)  
5. **Free intermediate tensors**  

```c
for (int epoch = 0; epoch < 2000; epoch++) {
    zf_adam_zero_grad(opt);

    Tensor *h1 = zf_dense_forward(fc1, X);
    Tensor *a1 = zf_tanh(h1);
    Tensor *pred = zf_dense_forward(fc2, a1);
    Tensor *loss = zf_loss_mse(pred, Y);

    zf_backward(loss);
    zf_adam_step(opt);

    // logging every 200 epochs
    if (epoch % 200 == 0) {
        float mse = compute_mse_raw(pred, Y);
        printf("Epoch %4d | loss=%.6f | mse=%.6f\n", epoch, loss->data[0], mse);
        print_param_stats("fc1.W", fc1->weights);
        // ... more prints
    }

    // release in reverse order
    zf_tensor_release(loss);
    zf_tensor_release(pred);
    zf_tensor_release(a1);
    zf_tensor_release(h1);
}
```

---

## Step 6: Clean Up / قدم ششم: پاک‌سازی

بعد از آموزش، همهٔ منابع رو آزاد می‌کنیم.  
After training, we free all resources.

```c
zf_adam_free(opt);
zf_dense_free(fc1);
zf_dense_free(fc2);
zf_tensor_release(X);
zf_tensor_release(Y);
```

---

## Expected Output / خروجی مورد انتظار

اگه همه چی رو درست انجام داده باشی، باید loss از حدود ۰.۵–۰.۸ شروع کنه و تا حدود ۰.۰۱–۰.۰۰۵ پایین بیاد. همچنین مقادیر وزن‌ها تغییر می‌کنن و گرادیان‌ها غیر صفر می‌شن. یه نمونه خروجی اول و وسط آموزش:

If you did everything right, the loss should start around 0.5–0.8 and drop to about 0.01–0.005. You'll see weight values changing and gradients being non-zero. Example output early and mid training:

```
Epoch    0 | loss=0.792384 | mse=0.792384
fc1.W: req=1 size=16 v0=2.421681 g0=-0.013810
...
Epoch  200 | loss=0.047286 | mse=0.047286
fc1.W: req=1 size=16 v0=2.376392 g0=0.004495
```

---

## Complete Code / کد کامل

همین کد رو می‌تونی با بقیهٔ فایل‌های کتابخونه کامپایل کنی. (مطمئن شو `adam.c` و `adam.h` رو هم به پروژه اضافه کردی.)

You can compile this code together with the rest of the library files. (Make sure you added `adam.c` and `adam.h` to the project.)

```c
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
    {
        return;
    }
    if (X->ndim != 2 || Y->ndim != 2)
    {
        return;
    }
    if (X->shape[0] != Y->shape[0])
    {
        return;
    }
    if (X->shape[1] != 1 || Y->shape[1] != 1)
    {
        return;
    }

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
    {
        return 0.0f;
    }
    if (pred->size != target->size)
    {
        return 0.0f;
    }
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
        {
            zf_dense_free(fc1);
        }
        if (fc2)
        {
            zf_dense_free(fc2);
        }
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
```

---

## Tips & Best Practices / نکته‌ها و روش‌های خوب

- **ترتیب آزادسازی حافظه خیلی مهمه.** همیشه تنسورهای میانی رو **بعد از** `backward` و `step` آزاد کن، و ترتیبشون رو برعکسِ ترتیبِ ایجادشون رعایت کن.  
- **`requires_grad` رو فقط برای پارامترها (وزن و بایاس) فعال کن.** داده‌ها و ورودی‌ها معمولاً بهش نیاز ندارن.  
- **از `Adam` با نرخ یادگیری کوچیک استفاده کن** (مثلاً ۰.۰۰۱) – خیلی زودتر از SGD معمولی همگرا میشه.  
- **همیشه یه sanity check اول بکن** – توی اپوک صفر `requires_grad` تنسورها رو چاپ کن تا مطمئن بشی گراف وصل شده.  
- اگه خواستی دقت مدل رو واقعی بسنجی، یه مجموعه تست جدا بساز و بعد از آموزش loss رو روش حساب کن.

---

امیدوارم این آموزش کمک کرده باشه که با زانافلو راحت کار کنی. سوالی بود، توی گیتهاب ایشو بذار یا باهامون حرف بزن!  
Hope this tutorial helps you get comfortable with Zanaflow. If you have questions, open an issue on GitHub or chat with us!
```