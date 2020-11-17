De-Doppler Analysis
===================

In this code, the following terminology is used:
    - Hit: Single strong narrowband signal in an observation.
    - Event: Strong narrowband signal that is associated with multiple hits
      across ON observations.

.. note::
   This code works for .dat files that were produced by seti_event.py
   after turboSETI version 0.8.2, and blimpy version 1.1.7 (~mid 2019). The 
   drift rates *before* that version were recorded with the incorrect sign
   and thus the drift rate sign would need to be flipped in the make_table 
   function.

Authors
-------
- Version 2.0 - Sofia Sheikh (ssheikhmsa@gmail.com) and Karen Perez (kip2105@columbia.edu)
- Version 1.0 - Emilio Enriquez (jeenriquez@gmail.com)

Find Event Pipeline
-------------------

.. automodule:: turbo_seti.find_event.find_event_pipeline
   :members:

Find Event
----------

.. automodule:: turbo_seti.find_event.find_event
   :members:

Plot Event Pipeline
-------------------

.. automodule:: turbo_seti.find_event.plot_event_pipeline
   :members:

Plot Event
----------

.. automodule:: turbo_seti.find_event.plot_event
   :members: