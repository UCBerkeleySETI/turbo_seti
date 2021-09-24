turboseti TUTORIAL
=============================

### Intro

Turbo_seti is a Python tool used and developed by SETI researchers at the Berkeley SETI Research Center. The alogorithm searches for narrow band signals which have a doppler drift, a feature expected to be seen from an alien source with a non-zero acceleration relative to our receivers on Earth.  Turbo_seti can be leveraged in 2 ways after installation:
1) Run the ```turboSETI``` executable at the bash (or Windows equivalent) command line in a terminal window.
2) Develop and run your own Python program which leverages the ```FindDoppler``` class and its functions.

This tutorial focuses on #2.

A basic introduction on installation and usage is given on turbo_seti's README.md on the GitHub site's highest-level directory. This tutorial intends to give a more comprehensive look at the usage of the tools contained in turbo_seti.

There are two ways of getting access to turbo_seti.

1) If you connected to the BL data center using the **ssh** utility, you can access a recent version by running source /opt/conda/init.sh in the terminal.  A recent version of turbo_seti should also be installed on the default Python3 kernel on the BL Jupyter Lab server.

2) You can install the latest up-to-date version of turbo_seti under your data center home directory (or on your own personal computer).

To install the latest version, there are 2 methods employed by folks:

1) The methodology described in the README.md on the front page of this Github site.  This could be a direct install, inside a ```Docker``` image, or as part of a ```venv``` virtual machine.

2) As part of a ```conda``` environment created following an Anaconda/Miniconda installation.

The following are some instructions for creating a new ```conda``` environment and adding it to a Jupyter Notebook:

First we'll need to set up an anaconda environment by running the following commands in the bash terminal (or Windows equivalent):

$ conda deactivate
$ conda create -n turboseti # The string "turboseti" after -n is arbitrary.  The name of your environment can be any of your choosing.
$ conda activate turboseti
$ conda install pip
Now let's install the required packages:

$ ~/.conda/envs/turboseti/bin/pip install git+https://github.com/UCBerkeleySETI/blimpy
~/.conda/envs/turboseti/bin/pip is the location of your pip. This may be different depending on your configuration.
blimpy is the file I/O for BL SETI. It will automatically gather all required packages for turboseti
$ ~/.conda/envs/turboseti/bin/pip install git+https://github.com/UCBerkeleySETI/turboseti

Now we need to install our environment as an IPython kernel, so we can use it in Jupyter:

$ conda install -c anaconda ipykernel
$ python -m ipykernel install --user --name=turboseti

A fresh version of turboseti is now installed in the kernel turboseti. You will have to restart Jupyter to see this kernel and be able to switch over (a quick refresh of the webpage should work).

