# To compile this with docker use:
# docker build --tag turbo_seti .
# Then to run it:
# docker run --rm -it turbo_seti
# To be able to access local disk on Mac OSX, you need to use Docker for Mac GUI
# and click on 'File sharing', then add your directory, e.g. /data/bl_pks
# Then to run it:
# docker run --rm -it -v /data/bl_pks:/mnt/data turbo_seti
# And if you want to access a port, you need to do a similar thing:
# docker run --rm -it -p 9876:9876 sigpyproc

# INSTALL BASE FROM KERN SUITE
FROM kernsuite/base:3
ARG DEBIAN_FRONTEND=noninteractive

ENV TERM xterm

######
# Do docker apt-installs
RUN docker-apt-install build-essential python3-setuptools python3-pip python3-tk pkg-config
RUN docker-apt-install curl wget make cmake git
RUN docker-apt-install libomp-dev
RUN docker-apt-install libhdf5-serial-dev libhdf5-dev libhdf5-cpp-11

#####
# Pip installation of python packages
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools wheel
RUN python3 -m pip install numpy
RUN python3 -m pip install pandas cython astropy matplotlib
RUN python3 -m pip install --only-binary=scipy scipy
RUN python3 -m pip install pytest
RUN python3 -m pip install git+https://github.com/ucberkeleyseti/blimpy

######
# HDF5 fixup
# Ubuntu 16.04 has a crazy hdf5 setup, needs some massaging, and extra flags to install h5py
RUN ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial.so /usr/lib/x86_64-linux-gnu/libhdf5.so
RUN ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so /usr/lib/x86_64-linux-gnu/libhdf5_hl.so

RUN CFLAGS=-I/usr/include/hdf5/serial pip install h5py



# Finally, install!
COPY . .
RUN python3 setup.py install



