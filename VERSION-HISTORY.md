This file is a version history of turbo_seti amendments, beginning with version 2.0.0.  Entries appear in version descending order (newest first, oldest last).
<br>
<br>

|    Date    | Version | Contents |
| :--: | :--: | :-- |
|  **==future==** | **TBD** | Add frequency channel masking capability. See issue #125. |
| 2021-02-25 | 2.0.9 | Fixed issue #195 - Stop find_event_pipeline() from crashing when there are no complex cadence matches. |
| | | Fixed issue #194 - Implemented complex cadence testing.
| | | Addressed issue #197 - Added file path ordering by header.tstart in {find,plot}_event_pipeline.py. |
| | | **Still outstanding**: Issue #89 (min_drift parameter is broken). |
| 2021-02-23 | 2.0.8.2 | Fixed issue #190 - Stop find_events() from crashing when a complex cadence has been specified. |
| 2021-02-20 | 2.0.8.1 | Address issue #188 - Enhance plot_event.py to handle both interactive and noninteractive matplotlib backends. |
| | | Removed references to numpy from setup.py. |
| 2021-01-30 | 2.0.8 | Added test/{test_fb_cases.py, fb_*} to implement checking against known valid DAT file results. |
| | | Added a --min_drift parameter to turboSETI (issue #178).  **Alas, issue #89 has been re-opened.** |
| | | Fixed the DAT file formatting and top hit numbering when multipatitioning with dask (issue #179). |
| 2021-01-20 | 2.0.7 | Fixed issues #135 & #150 (confirmed) by making code-structure in find_doppler.py search_coarse_channel() to be more like version 1.3.0. |
| 2021-01-19 | 2.0.6.6 | Fixed issue #169 - Fixed find_event_pipeline IndexError crash. |
| 2021-01-18 | 2.0.6.5 | Fixed issue #167 - LogWriter support for unattended testing on data centre compute nodes was enhanced to provide feedback concerning success/failure. |
| 2021-01-16 | 2.0.6.4 | Fixed issue #164 - Progress bar in dask partitioning is now OFF by default. |
| 2021-01-13 | 2.0.6.3 | Fixed issue #159 - Removed invalid bash script generation in setup.py. |
| 2021-01-12 | 2.0.6.2 | Fixed issue #157 - Logging enhancement in find_doppler.py. |
| 2021-01-09 | 2.0.6.1 | Fixed issue #154 - Enhanced test/test_pipelines.py. |
| 2021-01-05 | 2.0.6 | Fixed issue #152 - plot_event() by yanking the PR #82 code. Reprocussions to Parkes data? |
| 2021-01-04 | 2.0.5 | Rolling back previous fix to #141 (left open), hoping to fix issue #150 (related to issue #135). |
| 2020-12-31 | 2.0.4.1 | Fixed issue #141 which prevented searching in one of the drift block ranges. |
| 2020-12-24 | 2.0.4 | Added GitHub Actions Workflows for CI instead of Travis CI. |
| 2020-12-22 | 2.0.3 | Reverted changes made by PRs #121 and #113, hoping to fix issue #135. |
| 2020-12-21 | 2.0.2 | Amended `find_doppler.py`, `seti_event.py`, and `plot_event.py` to ressurect logging. See issue \#134. |
| 2020-12-20 | 2.0.1 | Amended `plot_event_pipeline.py` to accept a new filter_spec parameter. See issue \#127. |
| | | Amended `plot_event.py` to stop generating "RuntimeWarning: More than 20 figures have been opened".
| | | Default write-mode for DAT & LOG files is changed to "w" (replace). Append requires new optional `-a y` parameter.
| | | GPU-mode performance improvements.
| 2020-11-17 | 2.0.0 | Support NUMBA JIT compilation (CPU) and CUPY (NVIDIA GPU). |
| | | Made `turboSETI -n ...` work (set the number of coarse channels).
| | | No longer keeping Voyager test data in this repository.
| | | Fixed everal data-dependent crash-bug fixes.

