# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 13:21:43 2022

@author: blf20
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:49:47 2022

@author: Fogim
"""



import warnings
warnings.filterwarnings("ignore")


from datetime import datetime
now = datetime.now() # current date and time
from readPTU_FLIM import PTUreader
#from readPTU_FLIM_noheader import PTUreader
from scipy.stats import mode
import numpy as np
#import cupy as cp
import colorsys
import os
import tifffile
from ome_types import from_xml, from_tiff
from matplotlib import pyplot as plt
from PIL import Image
from numba import njit, jit
from numba.experimental import jitclass
from numba.types import uint16




#@njit
def get_lifetime_image(channel_number,timegating_start1,timegating_stop1,meas_resolution, flim_data_stack = np.array([[[[]]]])):
    
    #this function just gets mean arrival time after the peak of the pulse
    work_data  = flim_data_stack[:,:,channel_number,timegating_start1:timegating_stop1]
    bin_range = np.reshape(np.linspace(timegating_start1,timegating_stop1,timegating_stop1-timegating_start1),(1,1,timegating_stop1-timegating_start1))
    denom = np.sum(work_data,axis = 2)
    fast_flim = (np.sum(work_data*bin_range,axis = 2)/denom - timegating_start1)*(meas_resolution)
    
    fast_flim[np.isnan(fast_flim)] = 0
    
   
    
    return fast_flim
    
def stitch_im(im_stack):
    
    length = np.size(im_stack,2)
    stitched_im=im_stack[:,:,length-1,:]
    i = length - 2
    while i >= 0:
        
        stitched_im = np.concatenate((stitched_im, im_stack[:,:,i,:]), axis = 0)
        i = i-1
    
    return stitched_im
    



def plot_colorbar(im, title, max):
    
    fig, axs = plt.subplots(1)
    fig.suptitle(title)

    img = axs.imshow(im, vmax = max) 

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(img, cax=cbar_ax)
    

def save32bit(im, name):
    date_time = now.strftime("%m-%d-%Y, %H-%M-%S")
    x = 65536*im//np.amax(im)
    x = x.astype(np.uint16)
    x=x.T
    tifffile.imwrite('C:/Users/blf20/Documents/M3M/data/beam_extractor_output/proper/'+name+date_time+'.tif',x)
   
    
def saveOMETIFF(im, name):
    
    metadata={
         'axes': 'THYX',
         'PhysicalSizeX': 1,
         'PhysicalSizeXUnit': 'mm',
         'PhysicalSizeY': 6,
         'PhysicalSizeYUnit': 'mm',
         'TimeIncrement': 10,
         'TimeIncrementUnit': 'ps',
     }
    
    #metadata={'axes': 'THYX',}
    with tifffile.TiffWriter(name, ome=True) as tif:
        tif.write(np.expand_dims(np.moveaxis(im,(-1,0),(0,-1)), axis = 0), metadata=metadata)

    

    ome2 = from_tiff(name)
    
def initialise_paths():
    
    global read_path, write_path, file_num, filename, directory 
    
    read_path = 'Z:/User data/Billy Flanagan/2023/20230222tissueactin_zfine/'
    write_path = 'Z:/User data/Billy Flanagan/2023/python_ptu_processor_output/'

    directory_in_str = read_path
    directory = os.fsencode(directory_in_str)
    onlyfiles = next(os.walk(directory))[2] #directory is your directory path as string
    file_num = len(onlyfiles)

def slice_histo(full):
    fig, axs = plt.subplots(1)
    plt.plot(np.arange(np.size(full))/100, full)
    axs.set_yscale('log')
    plt.title('Fluorescence Decay')
    plt.xlabel('Time (ns)')
    plt.ylabel('Counts')
    
def lifetime_histo(flatflim):
   
    nbins = 200
    fig, axs = plt.subplots(1)
    y, x, z = plt.hist(flatflim, nbins)
    plt.title('Lifetime Histogram')
    plt.xlabel('Lifetime (s)')
    plt.ylabel('Frequency (#)')

    binsize = ((x.max()- x.min())/nbins)
    print ('Most occuring lifetime in s')
    mode_tau = y.argmax()*binsize
    print (mode_tau)


#%%

z=0
initialise_paths()

#check first file to get some parameters
filename = os.listdir(directory)[0].decode("utf-8") 
print(filename)
ptu_file  = PTUreader(read_path+filename, print_header_data = False)
flim_data_stack, intensity_image = ptu_file.get_flim_data_stack()

#save some basic variables
xpix = np.size(flim_data_stack,0)
ypix = np.size(flim_data_stack,1)
channels = np.size(flim_data_stack,2)

#create the full stacks
intensitystack = np.zeros((xpix,ypix, channels, file_num)) 
flim=np.zeros((xpix,ypix, channels)) 
flimstack=np.zeros((xpix,ypix, channels,file_num)) 


#goes through different files - corresponding to different z slices
for file in os.listdir(directory):
    
    if z >0:
        
        filename = os.fsdecode(file)
        print(filename)
        
        
        ptu_file  = PTUreader(read_path+filename, print_header_data = False)
        flim_data_stack, intensity_image = ptu_file.get_flim_data_stack()
    
    #here the flim data is stitched and saved as an ome tiff giving the histogram for each pixel for this z slice.
    #temp = stitch_im(flim_data_stack)
    #temp = np.array(temp, dtype='uint8')
    #saveOMETIFF(temp, write_path+'testtiff'+str(z)+'.ome.tif')
    
    bins = np.size(flim_data_stack,3)#bins is total number of bins in the TCSPC data
    bin_res = 12.7*1e-9/bins #just assume using 78Mhz must be careful here
    
    
    # #these variables are used to find the time at which the pulse occurs in the histogram
    full=np.zeros((channels,bins)) 
    peak=np.zeros(channels)

    
    # # these are the lifetime data for different channels
    for i in range(0, channels): 
        
          full[i,:] = np.sum(np.sum(flim_data_stack[:,:,i,:], axis = 0), axis = 0)
          peak[i] = np.argmax(full[i])#peak is the index at which max counts are registered over the whole FOV for a channel - start of pulse
          flim[:,:,i] = get_lifetime_image(i, int(peak[i]), bins, bin_res, flim_data_stack)
       
   
    
     
    flimstack[:,:,:,z] = flim[:,:,:] 
    intensitystack[:,:,:,z] = intensity_image[:,:,:] 
    
    z = z+1 
    
        

#%%

#now want to stitch lifetime and intensity stacks

stitching = True
#stitching = False

if stitching:
    
    stitched_intensity = stitch_im(intensitystack)
    stitched_flim      = stitch_im(flimstack)
else:
    stitched_intensity = (intensitystack)
    stitched_flim      = (flimstack)

#get some useful lifetime values


# #show example decay histogram for full slice 
slice_histo(full[3,:])



# #show a histogram of the lifetimes throughtout the whole data set
flatflim = stitched_flim.flatten()
flatflim = flatflim[np.argwhere(flatflim)]
lifetime_histo(flatflim)


# will leave doing the combined image for imageJ for now
#combined[:,:,:,:] = colorsys.hsv_to_rgb(stitched_flim[:,:,:]/5*mode_tau, 1, stitched_intensity[:,:,:]/np.amax(intensity_image))

#%%
#plot image for one slice
a = 0
#plot_colorbar(stitched_intensity[:,:,3,a], 'Full intensity image', np.amax(stitched_intensity[:,:,3,a]))
#plot_colorbar(stitched_flim[:,:,3,a], 'Full lifetime image', 3000*10**-12)# 10 picoseconds is measdesc resolution header value

#%%
#now save as files

save32bit(stitched_flim, 'flim_stack')
save32bit(stitched_intensity, 'intensity_stack')


#%%turn flim data stick into an ome-TIFF
# flim_1channel = flim_data_stack[:,:,3,:]
# saveOMETIFF(flim_1channel, write_path+'testtiff')



