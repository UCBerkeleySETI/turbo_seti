import os
import numpy as np
import cupy as cp

kernels_file = os.path.join(os.path.dirname(__file__), 'kernels.cu')

with open(kernels_file, 'r') as f:
    kernels = f.read()

_hitsearch_float64 = cp.RawKernel(kernels, 'hitsearch_float64')
_hitsearch_float32 = cp.RawKernel(kernels, 'hitsearch_float32')


def hitsearch(numBlocks, blockSize, call):
    r"""
    Performs hitsearch on the GPU with CUDA. Automatically chooses
    the right floating point precision based on the kernel configuration.

    Parameters
    ----------
    numBlocks : tuple
        CUDA Kernel number of blocks.
    blockSize : tuple
        CUDA Kernel block size.
    call : [int, ndarray, float, float, ndarray, ndarray, ndarray, float, float]
        Tuple of parameters required by `hitsearch`.

    """

    try:
        assert isinstance(call[0], int)
        assert isinstance(call[1], cp.ndarray)
        assert isinstance(call[2], float)
        assert isinstance(call[3], float)
        assert isinstance(call[4], cp.ndarray)
        assert isinstance(call[5], cp.ndarray)
        assert isinstance(call[6], cp.ndarray)
        assert isinstance(call[7], np.float32)
        assert isinstance(call[8], np.float32)
    except:
        raise ValueError("Check the `call` types of the `hitsearch` method.")

    if call[1].dtype == cp.float64:
        _hitsearch_float64(numBlocks, blockSize, call)
    if call[1].dtype == cp.float32:
        _hitsearch_float32(numBlocks, blockSize, call)
