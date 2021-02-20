This file is a version history of turbo_seti amendments, beginning with version 2.0.0.  Entries appear in version descending order (newest first, oldest last).
<br>
<br>
| Version | Contents |
| :--: | -- |
| 2.1.0 **future** | Add frequency channel masking capability. See issue #125. |
| 2.0.8.1 | Address issue #188 - Enhance plot_event.py to handle both interactive and noninteractive matplotlib backends |
| | Removed references to numpy from setup.py |
| 2.0.8 | Added test/{test_fb_cases.py, fb_*} to implement checking against known valid DAT file results |
| | Added a --min_drift parameter to turboSETI (issue #178) |
| | Fixed the min_drift parameter contribution to search_coarse_channel() (issue #89) |
| | Fixed the DAT file formatting and top hit numbering when multipatitioning with dask (issue #179) |
| 2.0.7 | Fixing code-structure in find_doppler.py search_coarse_channel() to be more like v1.3.0. |
| 2.0.6.6 | Fix issue #169 - Fixed find_event_pipeline IndexError crash |
| 2.0.6.5 | Fix issue #167 - log support for unattended testing on data centre compute nodes |
| 2.0.6.4 | Fix issue #164 - progress bar in dask partitioning should be OFF by default |
| 2.0.6.3 | Fix issue #159 - remove invalid bash script generation in setup.py |
| 2.0.6.2 | Fix issue #157 - logging enhancement in find_doppler.py |
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
