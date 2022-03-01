# TinyTorch :fire:

TinyTorch is a fork of [Minitorch](https://minitorch.github.io/) that I built out as part of Cornell Tech's Machine Learning Engineer course. Features include:
* Fast tensor data objects with a similar API to pytorch tensors
* Toolbox of tensor operations
* Automatic Differentiation
* Cuda and Optimized CPU support
* Basic pytorch.nn convolutions, pooling, and activation functions
* 75% test coverage using pytest, hypothesis (WIP)

**Setup**: Requires [poetry](https://python-poetry.org/) and python3.9 or 3.10
```bash
tinytorch> make venv
```

Running tests
```bash
# unit, style, and format test
tinytorch> make test

# ... or just unit tests (etc)
tinytorch> make test_unit
```

## Demos
Run LeNet CNN on MNIST dataset:
```bash
tinytorch> poetry run python tinytorch/examples/mnist_lenet.py
```
(still building out cli)
