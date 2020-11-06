import os 
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
    call : tuple
        Tuple of parameters required by `hitsearch`.

    """

    assert isinstance(call[2], float)

    if call[1].dtype == cp.float64:
        _hitsearch_float64(numBlocks, blockSize, call)
    if call[1].dtype == cp.float32:
        _hitsearch_float32(numBlocks, blockSize, call)