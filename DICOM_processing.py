# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:14:23 2022

@author: dbaltar
"""


import numpy as np
import pydicom as dicom
import os
import glob
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
import matplotlib.cm as cm
import vtk
import sys
import pylab
import pyvista as pv


def format_coord(x, y):
    col = int(x+0.5)
    row = int(y+0.5)
    if col>=0 and col<numcols and row>=0 and row<numrows:
        z = X[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)

def load_scan(path):
    # item1 = '*'
    slices = [os.path.join(data_path, item1) for item1 in os.listdir(path)]

    slices = [dicom.read_file(os.path.join(data_path, item1)) for item1 in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def sample_stack(stack, rows=15, cols=15, start_with=8, show_every=9):
    fig,ax = plt.subplots(rows,cols,figsize=[50,50])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

# -------------------- EXTRACT BODIES ----------------------------------

def itemnum(tup, num):
    print(tup[0][num], tup[1][num], tup[2][num])

def bodyid(r, percentilemin, percentilemax):
        # x = np.array(df[list(df)[2]])
        # y = np.array(df[list(df)[3]])
        # Create X, Y and Z?
        # r = np.array(df[list(df)[4]])
        total_points = np.size(r)
        # will need to generate something with the same shape as the input
        pol = np.zeros_like(r, dtype="int")
        # pointiduf = np.arange(total_points)
        
        

        # Cell size is used to determine the distance between points

        
        # rcutofflow = np.percentile(r,percentilemin)
        # rcutoffhigh = np.percentile(r,percentilemax)
        
        rcutofflow = percentilemin
        rcutoffhigh = percentilemax
         
        # Remove the values that are below the cutoff
        
        filtr = np.where((r >= rcutofflow) & (r <= rcutoffhigh))
        

        
        xf = np.array(filtr[0])
        yf = np.array(filtr[1])
        zf = np.array(filtr[2])
        
        # Create a polygon number = 0 with same size as filter
        
        polnum = 0
        pol = np.zeros(len(xf), dtype="int")
                
        pointid = np.arange(len(xf), dtype="int")
        
        seed_points = np.array([pointid[0]], dtype="int")
        
        # things to remove from id space
        
        idrem = pointid == pointid[0]
        idrem = idrem == False
        
        # oo arrays are what is left to select
        # print(len(xf), len(idrem))
        
        xfoo = xf[idrem]
        yfoo = yf[idrem]
        zfoo = zf[idrem]
        # idrem = pointidyo == pointid[0]
        # idrem = idrem == False
        pointidoo = pointid[idrem]
        # yfyo = yfyo[idrem]
        
        # identify new seeds
        
        while np.size(pointidoo) > 0:
            print(np.size(pointidoo), 'points remaining')
            
            new_seed_points = np.array([])
        #    print(polnum)
            
    
            for thing in seed_points:
        
                #coordinates of the seed
                idseed = pointid == thing
                
                xseed = xf[idseed]
                yseed = yf[idseed]
                zseed = zf[idseed]
                # find new seeds
                
                addx = pointidoo[np.where((abs(xfoo-xseed)) < 4)]
                addy = pointidoo[np.where((abs(yfoo-yseed)) < 4)]
                addz = pointidoo[np.where((abs(zfoo-zseed)) < 4)]
        # classify as seed points those that are close enough
        
                inter = np.intersect1d(addx,np.intersect1d(addy,addz))
                
                # inter = pointidoo[np.where(((abs(xfoo-xseed) + abs(yfoo-yseed) + abs(zfoo-zseed)) <= 6))]               
                # addy = pointidyo[np.where((yfyo > (yseed-cellsize)) & (yfyo < (yseed+cellsize)))]
                # addz = pointidzo[np.where((zfzo > (zseed-cellsize)) & (zfzo < (zseed+cellsize)))]
                # inter = np.intersect1d(addx,addy)
                if np.size(inter)>0:
                    for point in inter:
                    
                    #assign polygon
                    
                        pol[point] = polnum
                        
                        #remove from search zone
                        idrem = pointidoo == point
                        idrem = idrem == False
                        xfoo = xfoo[idrem]
                        yfoo = yfoo[idrem]
                        zfoo = zfoo[idrem]
                        pointidoo = pointidoo[idrem]
                    
                new_seed_points = np.append(new_seed_points, inter)
        #        print(np.size(new_seed_points))
            if np.size(new_seed_points)>0:
                
                seed_points = new_seed_points
                           
            else:
                
                # increase polygon number
                polnum +=1
                # create new seed, and assign new polygon number, remove from search zone
                seed_points = np.array([pointidoo[0]], dtype="int")
                idrem = pointidoo == pointidoo[0]
                idrem = idrem == False
                pointidoo = pointidoo[idrem]
                xfoo = xfoo[idrem]
                yfoo = yfoo[idrem]
                zfoo = zfoo[idrem]
                pol[np.where(pointid == seed_points)] = polnum
                
        blobsize = []
        
        numpol = np.max(pol)

        
        for ind in (np.arange(numpol)+1):
            blobsize.append((pol == ind).sum())
            
        return filtr, pol, blobsize

#--------------------- CODE -----------------------------
# init_notebook_mode(connected=True) 

data_path = r"C:\Users\dbaltar\Documents\JaimeMRI\images"
output_path = working_path = r"C:\Users\dbaltar\Documents\JaimeMRI\DBprocess"
g = glob(data_path + '/*')

# Print out the first 5 file names to verify we're in the right folder.
print ("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
print ('\n'.join(g[:5]))
id=0
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)

# ----------------------SAVE PROGRESS------------------------------------

np.save(output_path + "fullimages_%d.npy" % (id), imgs)

# ---------------- CREATE HISTOGRAM ----------------------------------------

file_used=output_path+"fullimages_%d.npy" % id
imgs_to_process = np.load(file_used).astype(np.float64) 

plt.hist(imgs_first.flatten(), bins=100, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

# ------------------- DISPLAY IMAGES --------------------------------------

id = 0
imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))


sample_stack(imgs_first, rows = 1, cols = 1, show_every = 1)


imgs_first = imgs[12:283:9]

test  = imgs_first.flatten()



ints = range(57,96,5)

fil = dict()
polygon = dict()
sizes = dict()

for ind in ints:

    fil[ind], polygon[ind], sizes[ind] = bodyid(imgs_first, np.percentile(test,ind), np.percentile(test,ind+8))

#------------------------ DISPLAY -------------------------------------------------------

# create array with zeros



# add 1 to polygons

# polygon = polygon -1

clusters = np.zeros_like(imgs_first, dtype = int)
# assign polygon numbers to plot

intslist = list(ints)

# for index in intslist[0]:
    
 # index = intslist[9]

for index in intslist:

    for ind in range(len(sizes[index])):
    
    # chose sizes 
        if sizes[index][ind] > 200:
            indexid = np.where(polygon[index] == (ind + 1))
        for item in indexid[0]:
            clusters[fil[index][0][item]][fil[index][1][item]][fil[index][2][item]] = ind + 1
        else:
            continue

# bods = np.array(fil[index]).transpose()

# pdata = pv.PolyData(bods)
# pdata['orig_sphere'] = np.arange(100)

# create many spheres from the point cloud


# sphere = pv.Sphere(radius=0.1, phi_resolution=50, theta_resolution=50)
# pc = pdata.glyph(scale=True, geom=sphere, orient=False)
# pc.plot(cmap='Reds')
data = pv.wrap(clusters)
data.spacing = tuple([6,1,1])
data.plot(volume=True)