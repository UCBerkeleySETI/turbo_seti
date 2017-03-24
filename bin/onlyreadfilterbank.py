import subprocess
import glob, os, sys, time
import numpy as np
import struct
from optparse import OptionParser
import socket

local_host = socket.gethostname()

if local_host == 'loui':
    #headerprogram="~/lofarsoft/src/code/release/share/pulsar/bin/header"   # In loui
    headerprogram="/cvmfs/softdrive.nl/emilioe/my_lofarsoft_build/release/share/pulsar/bin/"   # In loui
elif 'coma' in local_host:
    headerprogram="/vol/optcoma/lofarsoft/release/share/pulsar/bin/header" # In coma
elif 'locus' in local_host:
    headerprogram="/opt/cep/lus/daily/Tue/src/code/release/share/pulsar/bin/header" # In CEP2
elif 'bl' in local_host:
    headerprogram="/usr/local/sigproc/bin/header"   #
else:
    headerprogram="/cvmfs/softdrive.nl/emilioe/my_lofarsoft_build/release/share/pulsar/bin/"   #Grid

def downsample(data,axis,factor,keepdtype=True):
    if axis==0:
        if keepdtype:
            dtype=data.dtype
            return np.array(np.sum(data.reshape(data.shape[0]/factor,factor,data.shape[1]),1)/factor,dtype=dtype)
        else:
            return array(np.sum(data.reshape(data.shape[0]/factor,factor,data.shape[1]),1)/factor)
    else:
        print "downsampling of other axis not implemented yet. Requires DM shift"

class DataReader():
    """DataReader, read filterbank data from one file.   """

    def __init__(self,filename,blocksize=0,startsample=0):
        """ Initiate data reader and set basic parameters"""
        # read file header from filterbank and parse this data
        print "Reading file",filename
        try:
            proc=subprocess.Popen(['header',filename],stdout=subprocess.PIPE)
        except:
            proc=subprocess.Popen([headerprogram,filename],stdout=subprocess.PIPE)
        proc.wait()
        headerinforaw=proc.stdout.readlines()
        self.headerinfo=dict()
        for line in headerinforaw:
            self.headerinfo[line.split(':')[0].rstrip()]=':'.join(line.split(':')[1:]).strip('\n')
        # Header length, needed to read the raw datafile
        headerlen=int(self.headerinfo['Header size (bytes)'])
        # channels in the file (frequency unit)
        self.channels=int(self.headerinfo['Number of channels'])
        # total number of samples in the file (time unit)
        self.totsamples=int(self.headerinfo['Number of samples'])
        # beam number of the file
        self.beamnumber=int(self.headerinfo['Beam number'])
        #Number of bits per sample
        self.nbits=int(self.headerinfo['Number of bits per sample'])
        #Frequency of channel 1 (MHz)
        self.freq_init=float(self.headerinfo['Frequency of channel 1 (MHz)'])
        #Observation length (minutes)
        self.tot_time=float(self.headerinfo['Observation length (minutes)'])
        # set binary datatype, unsigned
        self.dtype='float'+str(self.nbits)
        # open datafile
        self.filfile=open(filename,'rb')
        # go to the start of the file
        self.filfile.seek(0)
        # read raw binary header
        self.hdrraw=self.filfile.read(headerlen)
        # set start sample
        self.startsample=int(startsample)
        # set start of data, at real length of header
        self.datastart=self.hdrraw.find('HEADER_END')+len('HEADER_END')+self.startsample*self.channels
        # go to start of data
        self.filfile.seek(self.datastart)
        # read data in sizes of this block
        if not blocksize:
            self.blocksize=self.totsamples
        else:
            self.blocksize=blocksize
        # blocknumbers are increased, so first block shoud be -1
        self.blocknumber=-1
        self.nrblocks=self.totsamples/self.blocksize
        # set LOFAR as telescope_id
        self.fixheader()


    def setstart(self,sampleoffset):
        self.datastart=self.hdrraw.find('HEADER_END')+len('HEADER_END')+(self.startsample+sampleoffset)*self.channels
        self.totsamples-=sampleoffset

    def read_all(self,reverse=True):
        """ read all the data.
            If reverse=True the x axis is flipped.
        """
        # go to start of the data
        self.filfile.seek(self.datastart)
        # read data into 2-D numpy array
#        data=np.fromfile(self.filfile,dtype=self.dtype).reshape(self.channels,self.blocksize,order='F')
        data=np.fromfile(self.filfile,dtype=self.dtype).reshape(self.blocksize, self.channels)
        if reverse:
            data = data[:,::-1]
        return data

    def read_row(self,rownumber,reverse=True):
        """ Read a block of data. The number of samples per row is set in self.channels
            If reverse=True the x axis is flipped.
        """

        # go to start of the row
        self.filfile.seek(self.datastart+self.channels*rownumber*(self.nbits/8))
        # read data into 2-D numpy array
        data=np.fromfile(self.filfile,count=self.channels,dtype=self.dtype).reshape(1, self.channels)
        if reverse:
            data = data[:,::-1]
        return data

    def read_rows(self,rownumber,n_rows,reverse=True):
        """ Read a block of data. The number of samples per row is set in self.channels
            If reverse=True the x axis is flipped.
        """

        # go to start of the row
        self.filfile.seek(self.datastart+self.channels*rownumber*(self.nbits/8))
        # read data into 2-D numpy array
        data=np.fromfile(self.filfile,count=self.channels*n_rows,dtype=self.dtype).reshape(n_rows, self.channels)
        if reverse:
            data = data[:,::-1]
        return data


    def fixheader(self):
        # set telescope_id to LOFAR
        self.hdrraw=self.hdrraw[0:self.datastart].replace('telescope_id\x00','telescope_id\x0b')






