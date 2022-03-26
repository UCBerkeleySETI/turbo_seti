This file is a version history of turbo_seti amendments, beginning with version 2.0.0.  Entries appear in version descending order (newest first, oldest last).
<br>
<br>

| `YYYY_MM_DD` | `Version` | `Contents` |
| :--: | :--: | :-- |
| 2022-03-24 | 2.2.0 |Introduced dat_filter utility (issue #303). |
| | | Enhance event analysis (plotSETI et al) to filter by SNR and drift rate (issue #303). |
| 2022-02-23 | 2.1.25 | Fix to the wrong drift rates when the number of time integrations is not a power of 2 (issue #302). |
| 2022-02-07 | 2.1.24 | Print versions of hdf5plugin and the HDF5 library (issue #299). |
| | | Enable the display of HDF5 library error messages which are inhibited by default (issue #299). |
| 2022-01-23 | 2.1.23 | Stop mangled path file names in data_handler.py & find_event_pipeline.py (issue #297). |
| | | Dependent on blimpy version >= 2.0.34 |
| 2021-12-05 | 2.1.22 | Add the ability to entertain .h5 files and .dat files in separate directories - part 2 (issue #294) |
| 2021-12-04 | 2.1.21 | Add the ability to entertain .h5 files and .dat files in separate directories - part 1 (issue #291) |
| 2021-12-01 | 2.1.20 | Add source code reference to the Read the Docs documentation. |
| 2021-11-29 | 2.1.19 | Fix to find_doppler.py for potentially lost signals in hitsearch. (issue #290) |
| 2021-11-19 | 2.1.18 | Fix to data_handler.py for handling of the NFPC header field from the new rawspec. (issue #285) |
| 2021-11-10 | 2.1.17 | Fix to find_event.py which was generating too many events & plots. (issue #283) |
| 2021-10-28 | 2.1.16 | Print a 2x3 data postage stamp when loading data for coarse channel 0 only. (issue #280, part 2) |
| 2021-10-23 | 2.1.15 | Print a 2x3 data postage stamp when loading data. (issue #280) |
| 2021-10-22 | 2.1.14 | Support new metadata field, NFPC. (issue #278). |
| 2021-09-13 | 2.1.13 | Make find_doppler easier to read and amend. (issue #274). |
| 2021-08-17 | 2.1.12 | Fix "AttributeError: module 'cupy' has no attribute '_core'". (issue #272). |
| 2021-08-12 | 2.1.11 | Specific MeerKAT files cause erratic behaviour in GPU mode (issue #270). |
| 2021-07-22 | 2.1.10 | The data_handler crashed during conversion of a 59 GiB filterbank file (issue #267). |
| 2021-07-22 | 2.1.9 | Performance improvement in gpu mode: default to single-precision (32-bit). |
| 2021-07-20 | 2.1.8 | Performance improvements and fix min_drift to prevent near-min-drift hits. |
| 2021-07-18 | 2.1.7 | Create a turbo_seti clone of blank_dc that is optional and uses a different strategy (issue #262). |
| 2021-07-15 | 2.1.6 | Calculate normalized value inside hitsearch kernel on GPU-mode. |
| 2021-07-16 | 2.1.5 | Failed to pass the gpu_id from find_doppler.py to data_handler.py (issue #254). |
| 2021-07-15 | 2.1.4 | Add GPU device selection with cli argument gpu_id. (issue #254). |
| 2021-07-15 | 2.1.3 | Diagnose out of range time steps with correct messages (issue #256). |
| | | Also, stop catching exceptions in seti_event.py which causes a cascade in tracebacks. |
| 2021-07-10 | 2.1.2 | Diagnose non-cadence sets of files in find_event_pipeline (issue #250). |
| 2021-07-09 | 2.1.1 | New turbo_seti utility: plotSETI. |
| 2021-07-04 | 2.1.0 | The function calc_freq_range uses hardcoded parameter values. These should instead be derived from the data. |
| | | See issue #231 for the full description and the resolution approach. |
| 2021-06-26 | 2.0.23 | Make data_handler.py provide useful info during exceptions (issue #243).
| | | Cleared up median vs mean confusion (issue #244).
| | | Stop using a Python3 reserved word for a function name (issue #245).
| 2021-06-14 | 2.0.22 | Pre-delete HDF5 file when input is a Filterbank file (.fil) (issue #241).
| 2021-06-11 | 2.0.21 | Log n_coarse_chan value when calculated by blimpy (issue #238).
| 2021-06-06 | 2.0.20 | Log drift_rate_resolution value (issue #236).
| 2021-04-21 | 2.0.19 | Change min_drift default to disallow near-zero drift.
| 2021-04-13 | 2.0.18 | Add GPU enabled Docker image build.
| 2021-04-07 | 2.0.17 | Fixed issue #230 - Added turbo_seti/find_event/plot_dat.py which makes a plot similar to the one produced by plot_candidate_events, but also includes the hits detected, in addition to the candidate signal. |
| 2021-04-03 | 2.0.16 | Fixed issue #225 - Ensure proper order of regression test execution. |
| | | Fixed issue #226 - Apparently useless plot_event.py code became a bug source in latest matplotlib. |
| | | Fixed issue #227 - Allow color & alpha selection in plot_event.py overlay_drift function. |
| | | Fixed issue #228 - test_pipelines_1 fails SNR comparison on MacOS. |
| 2021-03-20 | 2.0.15 | Fixed issue #205 - Reverse-engineered the original drift index files. |
| | | Fixed issue #218 - Replaced drift index file 8 (broken). |
| | | Fixed issue #94 - removed unused code from plot_event_pipeline.py and plot_event.py |
| 2021-03-10 | 2.0.14 | Fixed issue #213 - Doppler search dies when using GPU (string format issue). |
| | | Fixed issue #214 - Need some testing for plot_dir parameter of plot_event_pipeline. |
| 2021-03-09 | PR #212 | Support specification of an output directory for plotting at multiple levels in plot_event_pipeline.py and plot_event.py. |
| 2021-03-05 | 2.0.13 | Support very large data arrays.  See blimpy issue #180. |
| 2021-03-03 | 2.0.12 | Fixed issue #207 - flexible DAT line scanning in find_event.py read_dat(). |
| 2021-03-02 | 2.0.11 | Fixed issue #89 - min_drift & max_drift in find_doppler.py. |
| | | Fixed issue #162 - Announce turbo_seti and blimpy versions in use at start of Doppler search (find_doppler.py). |
| | | Fixed issue #200 - Cleanup/speedup of test_turbo_seti.py. |
| | | Fixed issue #201 - Created test_drift_rates for testing find_doppler.py min and max drift rates. |
| | | Fixed issue #202 - Amend plot_event.py to compute the blimpy Waterfall max_load parameter value for data arrays exceeding 1 GB in size. |
| | | Fixed issue #203 - Show turbo_seti version as part of turboSETI --help. Add a -v/--version parameter. |
| | | Fixed issue #204 - Stop the pipelines from loading data when they are only interested in Waterfall header fields. |
| 2021-02-25 | 2.0.10 | Addressed issue #197 - Added file path ordering by header.tstart in {find,plot}_event_pipeline.py. |
| 2021-02-25 | 2.0.9 | Fixed issue #195 - Stop find_event_pipeline() from crashing when there are no complex cadence matches. |
| | | Fixed issue #194 - Implemented complex cadence testing.
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

