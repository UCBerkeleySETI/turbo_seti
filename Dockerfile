ARG IMAGE=ubuntu:20.04
FROM ${IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

COPY . /turboseti
WORKDIR /turboseti

RUN cat dependencies.txt | xargs -n 1 apt install --no-install-recommends -y

RUN python3 -m pip install -U pip
RUN python3 -m pip install git+https://github.com/UCBerkeleySETI/blimpy
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install -r requirements_test.txt
RUN python3 setup.py install
RUN cd test && python3 download_test_data.py && cd ..
RUN cd test && bash run_tests.sh && cd ..

RUN find test -name "*.h5" -type f -delete
RUN find test -name "*.log" -type f -delete
RUN find test -name "*.dat" -type f -delete
RUN find test -name "*.fil" -type f -delete
RUN find test -name "*.png" -type f -delete
RUN find . -path '*/__pycache__*' -delete

WORKDIR /home
