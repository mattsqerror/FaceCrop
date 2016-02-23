# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 21:22:29 2016

@author: matt
"""

import glob
import os
import sys
import numpy as np
import scipy
import scipy.ndimage
import subprocess
import h5py

#%% Flags
CROP_FLAG = False

#%% get file dir/file information
timit_path = '/home/matt/Datasets/vidtimit'

osw_root = []
osw_flen = []
for root, dirs, files in os.walk(timit_path):
    
    if os.path.split(root)[-1][0] == 's':
        osw_root.append(root)
        osw_flen.append(len(files))
    #print dirs
    #print files

#%% create cropped faces jpegs 
if CROP_FLAG == True:
    os.chdir('/home/matt/python/FaceCrop')
    
    for oo in osw_root:
        print oo
        sys.stdout.flush()
        face_files = glob.glob(os.path.join(oo,'*'))
        for ff in face_files:
            subprocess.call(['./bin/CropFace', ff])
    
#%% extract data from cropped faces, there are 43 vidimit faces with 10 movies per face
NFRAMES=25  #number of frames per movie
NMOVIES=2   #number of movies per directory (reminder: the smallest movie only has 54 frames)
    
ii=0
vids = np.zeros([430*NMOVIES,NFRAMES,3,64*64])
for oo in osw_root:
    print oo
    sys.stdout.flush()
    face_files = glob.glob(os.path.join(oo,'*_face.jpg'))[:50]
    
    for kk in xrange(NMOVIES):    
        fac = face_files[kk*NFRAMES:(kk+1)*NFRAMES]
        for jj,ff in enumerate(fac):
            vids[ii,jj]=np.transpose(scipy.ndimage.imread(ff),[2,0,1]).reshape(3,64*64)
        ii+=1

#%% save faces with h5py, xx = data, yy = people labels
xx = vids.reshape(430*NMOVIES,NFRAMES,3*64*64)
yy = np.kron(np.arange(43),np.ones(10*NMOVIES))     # won't save these, too simple

faces = h5py.File('/home/matt/Datasets/faces.h5', 'w')
faces.create_dataset('faces',data=xx)
faces.close()




