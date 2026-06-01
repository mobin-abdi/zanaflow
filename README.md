# Zanaflow

**Zanaflow** is a modular, lightweight deep learning framework implemented from scratch in C. 

This project aims to provide a clear, educational, and clean foundation for understanding how neural network components, tensor operations, and optimization algorithms work under the hood, without relying on heavy external dependencies.

> **Status:** Work In Progress (WIP). This project is under active development.

## Features
Currently, Zanaflow provides the building blocks for creating and training simple neural networks:
*   **Tensor Operations:** Multi-dimensional array handling with basic arithmetic.
*   **Neural Network Layers:** Fully implemented `Dense` layer with backward pass support.
*   **Optimizers:** Stochastic Gradient Descent (SGD) implementation.
*   **Activations:** Built-in support for ReLU, Leaky ReLU, Tanh, and Softmax.
*   **Loss Functions:** MSE Loss implementation.
*   **Modular Architecture:** Designed with separation of concerns between layers, optimizers, and tensors.

## Quick Start
You can easily include the framework and start building.
```c
#include <zanaflow.h>

int main() {
// Example: Create a simple tensor and run a operation
Tensor* t = tensor_create((size_t[]){2, 2}, 4);
tensor_fill_random_uniform(t, -1.0f, 1.0f);

// ... define layers, perform forward pass ...

tensor_free(t);
return 0;
}
```
## Project Roadmap
- [x] Tensor Core & Basic Ops
- [x] Dense Layers & SGD Optimizer
- [x] Activation Functions
- [x] Autograd Engine: Implementing the computation graph for automatic differentiation.
- [x] More Optimizers: Adam, RMSProp.
- [ ] Advanced Layers: Conv2D, Dropout, Pooling.
- [ ] Documentation: Generating full Doxygen API reference.

## Project Structure
```
zanaflow/
├── include/zanaflow/    # Public headers (The API)
├── src/                 # Implementation logic
├── examples/            # Usage examples
├── tests/               # Unit tests
└── docs/                # Architecture & guides
```

## Contributing
Since this is an educational project, contributions are welcome! Whether it’s adding new layers, improving tensor performance, or writing unit tests, feel free to open a Pull Request.

## License
Distributed under the Apache License 2.0. See LICENSE for more information.
