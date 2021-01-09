This file is a version history of turbo_seti amendments, beginning with version 2.0.0.  Entries appear in version descending order (newest first, oldest last).
<br>
<br>
| Version | Contents |
| :--: | -- |
| 2.1.0 **coming soon** | Add frequency channel masking capability. See issue #125. |
| 2.0.6.1 | Fix issue #154 - test/test_pipelines.py. |
| 2.0.6 | Fix issue #152 - plot_event(). |
| 2.0.5 | Fix issue #150 by rolling back previous fix to #141 (left open). |
| 2.0.4.1 | Fix issue #141 which prevented searching in one of the drift block ranges. |
| 2.0.4 | Add GitHub Actions Workflows for CI instead of Travis CI. |
| 2.0.3 | Fix issue #135 by reverting changes made by PR #121 and #113. |
| 2.0.2 | Amended `find_doppler.py`, `seti_event.py`, and `plot_event.py` to ressurect logging. See issue \#134. |
| 2.0.1 | Amended `plot_event_pipeline.py` to accept a new filter_spec parameter. See issue \#127. |
| | Amended `plot_event.py` to stop generating "RuntimeWarning: More than 20 figures have been opened".
| | Default write-mode for DAT & LOG files is changed to "w" (replace). Append requires new optional `-a y` parameter.
| | GPU-mode performance improvements.
| 2.0.0 | Support NUMBA JIT compilation (CPU) and CUPY (NVIDIA GPU). |
| | Made `turboSETI -n ...` work (set the number of coarse channels).
| | No longer keeping Voyager test data in this repository.
| | Several data-dependent crash-bug fixes.
