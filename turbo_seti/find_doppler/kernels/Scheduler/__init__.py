from concurrent.futures import ThreadPoolExecutor


class Scheduler:
    def __init__(self, method, params, n_futures=2, n_workers=2):
        self.n_workers = n_workers
        self.n_futures = n_futures
        self.method = method
        self.params = params
        
        self._init_threads()
        self._update_futures()
        
    @staticmethod
    def _batch(iterable, n):
        return iterable[:min(n, len(iterable))]
        
    def _init_threads(self):
        self.futures = []
        self.client = ThreadPoolExecutor(self.n_workers)
        
    def _update_futures(self):
        n = self.n_futures - len(self.futures)
        for p in self._batch(self.params, n):
            self._submit_future(p)
        
    def _submit_future(self, p):
        call = (self.method, *p)
        future = self.client.submit(*call)
        self.futures.append(future)
        self.params.remove(p)
        
    def get(self):
        self._update_futures()
        for f in self.futures:
            result = f.result()
            self.futures.remove(f)
            return result