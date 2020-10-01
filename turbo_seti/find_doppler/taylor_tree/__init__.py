import importlib

class TaylorTree():

    def __init__(self, core_id="numba"):
        self.core_id = core_id
        self._load_modules()

    def _load_modules(self):
        base_lib = "turbo_seti.find_doppler.taylor_tree"
        if self.core_id == "cuda":
            self.core = importlib.import_module(base_lib + '._core_cuda')
        if self.core_id == "numba":
            self.core = importlib.import_module(base_lib + '._core_numba')
