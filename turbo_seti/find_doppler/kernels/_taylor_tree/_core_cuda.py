import cupy as cp
import numpy as np

# Type pairs that we support
TYPE_PAIRS = {"float": cp.float32, "double": cp.float64}

# Cuda kernels for the flt function to use.
# Based on the original C code by Franklin Antonio, available at
#   https://github.com/UCBerkeleySETI/dedopplerperf/blob/main/CudaTaylor5demo.cu
# It does one round of the Taylor tree algorithm, calculating the sums of length-2^(x+1) paths
# from the sums of length-2^x paths.
CODE = r"""
template<typename T>
__global__ void taylor(const T* A, T* B, int kmin, int kmax, int set_size, int n_time, int n_freq) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int k = kmin + tid;
  bool worker = (k >= kmin) && (k < kmax) && set_size <= n_time;
  if (!worker) {
    return;
  }
  for (int j = 0; j < n_time; j += set_size) {
    for (int j0 = set_size - 1; j0 >= 0; j0--) {
      int j1 = j0 / 2;
      int j2 = j1 + set_size / 2;
      int j3 = (j0 + 1) / 2;
      if (k + j3 < kmax) {
        B[(j + j0) * n_freq + k] = A[(j + j1) * n_freq + k] + A[(j + j2) * n_freq + k + j3];
      }
    }
  }
}
"""
C_TYPES = TYPE_PAIRS.keys()
NAME_EXPS = [f"taylor<{t}>" for t in C_TYPES]
MODULE = cp.RawModule(code=CODE, options=("-std=c++11",), name_expressions=NAME_EXPS)
KERNELS = {}
for c_type, name_exp in zip(C_TYPES, NAME_EXPS):
    KERNELS[c_type] = MODULE.get_function(name_exp)


def flt(array, n_time):
    """
    Taylor-tree-sum the data in array.

    array should be a 1-dimensional cupy array. If reshaped into two dimensions, the
    data would be indexed so that array[time][freq] stores the data at a particular time
    and frequency. So, the same way h5 files are typically stored.

    n_time is the number of timesteps in the data.

    The algorithm uses one scratch buffer, and in each step of the loop, it calculates
    sums from one buffer and puts the output in the other. Thus, the drift sums we are looking
    for may end up either in the original buffer, or in the scratch buffer. This method
    returns whichever buffer is the one to use, and we leave the other one for cupy to clean up.
    """
    taylor_kernel = None
    for c_type, py_type in TYPE_PAIRS.items():
        if py_type == array.dtype:
            taylor_kernel = KERNELS[c_type]
            break
    else:
        raise RuntimeError(
            f"we have no GPU taylor kernel for the numerical type: {array.dtype}"
        )

    assert len(array) % n_time == 0
    n_freq = len(array) // n_time
    buf = cp.zeros_like(array)

    # Cuda params
    block_size = 1024
    grid_size = (n_freq + block_size - 1) // block_size

    set_size = 2
    while set_size <= n_time:
        taylor_kernel(
            (grid_size,),
            (block_size,),
            (array, buf, 0, n_freq, set_size, n_time, n_freq),
        )
        array, buf = buf, array
        set_size *= 2

    return array
