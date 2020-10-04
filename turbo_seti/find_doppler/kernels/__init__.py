import importlib

class Kernels():

    def __init__(self, core_id="numba"):
        self.core_id = core_id
        self.base_lib = "turbo_seti.find_doppler.kernels"

        self._load_taylor_tree()
        self._load_hitsearch()

    def _load_taylor_tree(self):
        if self.core_id == "cuda":
            self.tt = importlib.import_module(self.base_lib + '._taylor_tree._core_cuda')
        if self.core_id == "numba":
            self.tt = importlib.import_module(self.base_lib + '._taylor_tree._core_numba')

    def _load_hitsearch(self):
        if self.core_id == "cuda":
            self.hitsearch = importlib.import_module(self.base_lib + '._hitsearch').hitsearch
