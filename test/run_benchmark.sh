#!/bin/bash

FILE=blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.fil
if test ! -f "$FILE"; then
    echo "$FILE not found downloading it..."
    URL=http://blpd0.ssl.berkeley.edu/voyager_2bit/$FILE

    if hash aria2c 2>/dev/null
    then
        aria2c -x 8 -s 8 $URL
    else
        wget $URL
    fi
fi

FILE=blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.fil
if test ! -f "$FILE"; then
    echo "$FILE not found downloading it..."
    URL=http://blpd0.ssl.berkeley.edu/voyager_8bit/$FILE

    if hash aria2c 2>/dev/null
    then
        aria2c -x 8 -s 8 $URL
    else
        wget $URL
    fi
fi

FILE=blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000
if test ! -f "$FILE.h5"; then
    echo "Generating $FILE"
    fil2h5 $FILE.fil
fi

FILE=blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000
if test ! -f "$FILE.h5"; then
    echo "Generating $FILE"
    fil2h5 $FILE.fil
fi

echo "====> [BENCHMARK] GPU DOUBLE PRECISION"
turboSETI Voyager1.single_coarse.fine_res.h5 -g y -S n -P n
turboSETI blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.h5  -g y -S n -P n
turboSETI blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.h5  -g y -S n -P n

echo "====> [BENCHMARK] GPU SINGLE PRECISION"
turboSETI Voyager1.single_coarse.fine_res.h5 -g y -S y -P n
turboSETI blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.h5  -g y -S y -P n
turboSETI blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.h5  -g y -S y -P n

echo "====> [BENCHMARK] CPU DOUBLE PRECISION"
turboSETI Voyager1.single_coarse.fine_res.h5 -g n -S n -P n
turboSETI blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.h5  -g n -S n -P n
turboSETI blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.h5  -g n -S n -P n

echo "====> [BENCHMARK] CPU SINGLE PRECISION"
turboSETI Voyager1.single_coarse.fine_res.h5 -g n -S y -P n
turboSETI blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.h5  -g n -S y -P n
turboSETI blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.h5  -g n -S y -P n