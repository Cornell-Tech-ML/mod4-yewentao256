# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.

Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```bash
pip install python-mnist
mnist_get_data.sh
```

* Tests:

```bash
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

```bash
minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py
```

## Task 4.4 - Bonus Convolution

Code: [cuda_conv.py](minitorch/cuda_conv.py)

Test case: [test_cuda_conv.py](tests/test_cuda_conv.py)

Result:

```bash
PS C:\Users\Peter\Desktop\mod4-yewentao256> pytest .\tests\test_cuda_conv.py
======================================================== test session starts ========================================================
platform win32 -- Python 3.11.5, pytest-8.3.2, pluggy-1.5.0
rootdir: C:\Users\Peter\Desktop\mod4-yewentao256
configfile: pyproject.toml
plugins: hypothesis-6.54.0, env-1.1.3
collected 4 items

tests\test_cuda_conv.py ....                                                                                                   [100%]

========================================================= warnings summary ==========================================================
tests/test_cuda_conv.py::test_conv1d_cuda_simple
tests/test_cuda_conv.py::test_conv2d_cuda_simple
  C:\Users\Peter\AppData\Local\Programs\Python\Python311\Lib\site-packages\numba\cuda\dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_cuda_conv.py::test_conv1d_cuda_simple
tests/test_cuda_conv.py::test_conv1d_cuda_random
tests/test_cuda_conv.py::test_conv2d_cuda_simple
tests/test_cuda_conv.py::test_conv2d_cuda_random
  C:\Users\Peter\AppData\Local\Programs\Python\Python311\Lib\site-packages\numba\cuda\cudadrv\devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
    warn(NumbaPerformanceWarning(msg))

tests/test_cuda_conv.py::test_conv1d_cuda_random
  C:\Users\Peter\AppData\Local\Programs\Python\Python311\Lib\site-packages\numba\cuda\dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=================================================== 4 passed, 7 warnings in 5.56s ===================================================
```

## Task 4.5

### Mnist

[Logs](mnist.txt)
