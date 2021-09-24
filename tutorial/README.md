TURBO_SETI TUTORIAL
=============================

### Intro

The turbo_seti package is a Python tool, used and developed by SETI researchers at the Berkeley SETI Research Center. The alogorithm of the code in the find_doppler directory searches for narrow band signals which have a doppler drift, a feature expected from an alien source with a non-zero acceleration relative to our receivers on Earth. 

We'll search some HDF5 files that have been condensed to a single coarse channel and are routinely used for testing the code. They are located here: http://blpd0.ssl.berkeley.edu/parkes_testing/ (total download size of 294 MB).

Typical SETI searches, such as described by https://arxiv.org/pdf/1906.07750.pdf, have used drift rates of up to ±4 Hz and a minimum signal-to-noise ratio of 10. We'll adapt those parameters for these tutorials.

There are 2 methods of executing a turbo_seti search after package installation:
1) Run the ```turboSETI``` executable at the bash (or Windows equivalent) command line in a terminal window.
2) Develop and run your own Python program which leverages the ```FindDoppler``` class and its functions.

Contained herein are 2 tutorials that focus on method #2.  Once either of these tutorials have been successfully completed, then the use of method #1 will be straight-forward (run ```turboSETI -h``` for details).

The two tutorials are as follows:
* ```tutorial_1.ipynb``` - This exposes details of the event pipelines after the search completes and is therefore more complex.
* ```tutorial_2.ipynb``` - A simpler approach (same results) for analyzing candidate events.

Both tutorials require the execution of an initialization noyebook before using them: ```initialise.ipynb```.  This will download the set of 6 Parkes HDF5 files into a directory called "turboseti" under the user's home directory.

After the files are downloaded, install the latest version of turbo_seti, in one of two ways:

1) The methodology described in the README.md on the front page of this Github site.  This could be a direct install, inside a ```Docker``` image, or as part of a ```venv``` virtual machine.

2) As part of a ```conda``` environment created following an Anaconda/Miniconda installation.

If you followed installation method #1, you are ready to execute either tutorial.

If you intend to follow installation method #2, then the remainder of this README.md describes how to create a new ```conda``` environment and add it to a Jupyter Notebook.

First we'll need to set up an anaconda environment by running the following commands in the bash terminal (or Windows equivalent):
```
$ conda deactivate
$ conda create -n turboseti # The string "turboseti" after -n is arbitrary.  The name of your environment can be any of your choosing.
$ conda activate turboseti
$ conda install pip
```

Now let's install the required packages:
```
$ ~/.conda/envs/turboseti/bin/pip install git+https://github.com/UCBerkeleySETI/blimpy
~/.conda/envs/turboseti/bin/pip is the location of your pip. This may be different depending on your configuration.
blimpy is the file I/O for BL SETI. It will automatically gather all required packages for turboseti
$ ~/.conda/envs/turboseti/bin/pip install git+https://github.com/UCBerkeleySETI/turboseti
```

Now we need to install into the turboseti environment as an IPython kernel, so we can use it in Jupyter:
```
$ conda install -c anaconda ipykernel
$ python -m ipykernel install --user --name=turboseti
```
A ready-to-use version of turboseti is now installed. You will have to restart Jupyter to see this kernel and be able to switch over (a quick refresh of the webpage should work).
