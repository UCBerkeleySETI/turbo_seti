De-Doppler Search
=================

Find Doppler
------------

.. automodule:: turbo_seti.find_doppler.find_doppler
   :members:

File Writers
------------

.. automodule:: turbo_seti.find_doppler.file_writers
   :members:

Data Handler
------------

.. automodule:: turbo_seti.find_doppler.data_handler
   :members:

Kernels
-------

.. automodule:: turbo_seti.find_doppler.kernels
   :members:

   Hitsearch
   ----------
   This kernel implements a GPU accelerated version of the :func:`~turbo_seti.find_doppler.find_doppler.hitsearch`
   method written as a RAW CUDA kernel.

   .. automodule:: turbo_seti.find_doppler.kernels._hitsearch
      :members:

   De-Doppler
   -----------
   This kernel implements a slightly modified version of the Taylor Tree algorithm
   `published <http://articles.adsabs.harvard.edu/pdf/1974A%26AS...15..367T>`_ by J.H. Taylor in 1974.

      1. This GPU implementation is based on `Cupy <https://cupy.dev/>`_ array library accelerated with CUDA and ROCm.

      .. automodule:: turbo_seti.find_doppler.kernels._taylor_tree._core_cuda
         :members:

      2. This CPU implementation is based on `Numba <https://numba.pydata.org/>`_ Just-In-Time compilation.

      .. automodule:: turbo_seti.find_doppler.kernels._taylor_tree._core_numba
         :members:

Helper Functions
----------------

.. automodule:: turbo_seti.find_doppler.helper_functions
    :members:
