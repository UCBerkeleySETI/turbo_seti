import importlib


class Kernels():

    def __init__(self, gpu_backend, precision=2):
        self._gpu_backend = gpu_backend
        self._precision = precision
        self._base_lib = "turbo_seti.find_doppler.kernels"

        self._load_base()
        self._load_taylor_tree()
        self._load_hitsearch()

    def _load_precision(self):
        if self._precision == 2:
            return self.xp.float64
        if self._precision == 1:
            return self.xp.float32
        if self._precision == 0:
            return self.xp.float16

        raise ValueError('Invalid float precision.')

    def _load_base(self):
        if self._gpu_backend:
            self.xp = importlib.import_module("cupy")
            self.np = importlib.import_module("numpy")
        else:
            self.xp = importlib.import_module("numpy")
            self.np = self.xp

        self.float_type = self._load_precision()

    def _load_taylor_tree(self):
        if self._gpu_backend:
            self.tt = importlib.import_module(self._base_lib + '._taylor_tree._core_cuda')
        else:
            self.tt = importlib.import_module(self._base_lib + '._taylor_tree._core_numba')

    def _load_hitsearch(self):
        if self._gpu_backend:
            self.hitsearch = importlib.import_module(self._base_lib + '._hitsearch').hitsearch
