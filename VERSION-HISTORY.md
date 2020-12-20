This file is a version history of turbo_seti amendments, beginning with version 2.0.0.  Entries appear in version descending order (newest first, oldest last).
<br>
<br>
| Version | Contents |
| :--: | -- |
| 2.0.1  | Amended `plot_event_pipeline.py` to accept a new filter_spec parameter. See issue \#127. |
| | Amended `plot_event.py` to stop generating "RuntimeWarning: More than 20 figures have been opened".
| | Default write-mode for DAT & LOG files is changed to "w" (replace). Append requires new optional `-a y` parameter.
| | GPU-mode performance improvements.
| | Restored console logging.
| 2.0.0  | Support NUMBA JIT compilation (CPU) and CUPY (NVIDIA GPU). |
| | Made `turboSETI -n ...` work (set the number of coarse channels).
| | No longer keeping Voyager test data in this repository.
| | Several data-dependent crash-bug fixes.
