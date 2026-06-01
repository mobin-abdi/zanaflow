# ZanaFlow

**ZanaFlow** is a modular, lightweight deep learning framework implemented **from scratch in C** — focused on clarity, clean abstractions, and a practical Autograd core.

**زانا‌فلو** یک فریمورک سبک و ماژولار یادگیری عمیق است که **از صفر با C** پیاده‌سازی شده؛ هدفش شفافیت، معماری تمیز و یک هسته‌ی Autograd کاربردی است.

> **Status:** v0.1.x — early-stage, API may change.  
> **وضعیت:** نسخه‌های اولیه؛ احتمال تغییر API وجود دارد.

---

## Why ZanaFlow? | چرا ZanaFlow؟

**EN**
ZanaFlow is built for:
- Understanding neural networks “under the hood” (tensors → ops → graph → gradients)
- Writing fast, dependency-minimal C code
- Having a small, hackable codebase for experiments and learning

**FA**
زانا‌فلو ساخته شده برای:
- فهم دقیق شبکه‌های عصبی از داخل (تنسور → عملیات → گراف → گرادیان)
- کدنویسی سریع و کم‌وابستگی در C
- داشتن کدبیس کوچک و قابل هک برای تجربه و یادگیری

---

## Features | قابلیت‌ها

**Core**
- **Tensor** (float32) with explicit memory management
- **Autograd engine** (dynamic computation graph + backprop)
- **Deterministic examples/tests** (to validate correctness)

**NN**
- `Dense` (Linear) layer with backward
- Activations: ReLU, Leaky ReLU, Tanh, Softmax *(as available in repo)*  
- Loss: MSE *(as available in repo)*

**Optimizers**
- SGD
- Adam, RMSProp *(if implemented in your repo; otherwise remove)*

**Data**
- CSV loader (`zf_csv_load_f32`) for small datasets & examples  
  (Note: large datasets will move to a streaming/binary record loader.)

---

## Installation / Build | نصب / بیلد

**EN**
ZanaFlow is a C library. Compile it together with your project or build a static library.

**FA**
زانا‌فلو یک کتابخانه C است. می‌توانید آن را همراه پروژه‌تان کامپایل کنید یا به صورت استاتیک بسازید.

### Build example (single command)
> Adjust paths based on your repo layout.

```bash
cc -O2 -Wall -Wextra \
  -I./include \
  examples/mlp_example.c \
  src/*.c \
  -lm -o mlp
```

### Run:
```bash
./mlp
```

---

### Quick Start | شروع سریع
#### This snippet is intentionally minimal. For full training loops, see examples/.

```c
#include <zanaflow.h>

int main(void) {
// Create a 2x2 tensor (example — adjust to your actual API)
// Prefer using the official zf_* APIs to keep consistency.
//
// Tensor *t = zf_tensor_create((int[]){2,2}, 2);

return 0;
}
```

---

##### ✅ Tip: Keep inputs/targets with requires_grad = 0 to reduce memory usage.

---

### Examples | مثال‌ها

- examples/mlp_examples.c — MLP learns sin(x)
- tests/ — unit tests for loader / ops (as available)

---

### Project Roadmap | نقشه راه

- [x] Tensor core + basic ops
- [x] Dense + activations + MSE
- [x] Autograd (dynamic graph)
- [x] Optimizers: SGD (+ Adam/RMSProp if present)
- [x] CSV loader (in-memory) with error reporting
- [ ] Binary/Streaming data loader (record-based)
- [ ] Advanced layers: Conv2D, Dropout, Pooling
- [ ] Doxygen: full API reference
- [ ] More tests + CI (GitHub Actions)

---

### Project Structure | ساختار پروژه

```text
zanaflow/
├── include/zanaflow/     # Public headers (API)
├── src/                  # Implementation
├── examples/             # Examples (training loops / demos)
├── tests/                # Unit tests
└── docs/                 # Guides & architecture notes
```

---

### Contributing | مشارکت

**EN**
PRs are welcome — especially tests, examples, and docs.
If you want to add a feature, please open an Issue first to align on API design.

**FA**
Pull Request آزاد است—به‌خصوص تست‌ها، مثال‌ها و مستندات.
برای فیچر جدید بهتر است اول Issue باز کنید تا روی طراحی API هماهنگ شویم.

*شما می توانید در تلگرام هم با من در ارتباط باشید*
*you can connect me in Telegram*

---

### License | لایسنس

*Apache-2.0 — see LICENSE.*