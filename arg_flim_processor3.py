# -*- coding: utf-8 -*-
"""
Updated script to save TIFF files in the format:
A-1 - FOVXXXXX_YYYY-MM-DD.tiff
"""

import argparse
import os
from datetime import datetime
import numpy as np
import tifffile
from ome_types import from_tiff, to_xml
from readPTU_flim_noheader_test import PTUreader

# Function to compute mean arrival time
def compute_mean_arrival_time(fullstack):
    x, y, T, z = fullstack.shape
    lifetime_bins = np.arange(T).reshape((1, 1, T, 1))
    weighted_sum = np.sum(fullstack * lifetime_bins, axis=2)
    total_intensity = np.sum(fullstack, axis=2)
    mean_arrival_time = np.where(total_intensity > 0, weighted_sum / total_intensity, 0)
    return mean_arrival_time

# Create folder with a suffix if needed
def create_folder_with_suffix(base_path):
    counter = 1
    while os.path.exists(base_path):
        base_path = f"{base_path.rstrip('/')}_{counter}"
        counter += 1
    os.makedirs(base_path)
    print(f"Directory created: {base_path}")
    return base_path

# Create subfolder within a directory
def create_subfolder(parent_folder, subfolder_name):
    subfolder_path = os.path.join(parent_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)
    print(f"Subfolder created: {subfolder_path}")
    return subfolder_path

# Initialize paths
def initialise_paths(writepath):
    global read_path, file_num, directory
    read_path = 'C:/Program Files/Micro-Manager-2.0/tttr_raws/'
    directory = os.fsencode(read_path)
    onlyfiles = next(os.walk(directory))[2]
    file_num = len(onlyfiles)

# Function to stitch images
def stitch_im_new1(im_stack):
    s = im_stack    
    st1 = np.concatenate((s[:,:,0,:], s[:,:,1,:], s[:,:,2,:]), axis=0)
    st2 = np.concatenate((s[:,:,3,:], s[:,:,4,:], s[:,:,5,:]), axis=0)
    st = np.concatenate((st1, st2), axis=1)
    return st

# Save OME TIFF files
def save_FLIM_ome_tiff(im, name, met, binsize, write_path):
    metadata = {'axes': 'THYX'}
    im = np.expand_dims(im, axis=0)
    im = np.transpose(im, (0, 3, 2, 1))

    if met == 1:
        ome2 = from_tiff(write_path + 'test.tif')
        ome2.structured_annotations[0].value.any_elements[0].children[0].attributes = {
            'Type': 'lifetime', 'Start': '0', 'End': str(12600 - binsize), 'Step': str(binsize)}
        metadata_xml = to_xml(ome2).replace('\n', '')

        tifffile.imwrite(name, im, description=metadata_xml, metadata=metadata)
    else:
        with tifffile.TiffWriter(name, ome=True, bigtiff=True) as tif:
            tif.write(im, metadata=metadata)

def extract_datetime_from_filename(filename):
    """ Extracts the date & time in YYYY-MM-DD_HH-MM-SS format from filenames like '13-02-2025_14-10-33_tttr.out' """
    try:
        date_time_part = filename.split('_tttr')[0]  # Extract '13-02-2025_14-10-33'
        date_part, time_part = date_time_part.split('_')  # Split into date and time
        day, month, year = date_part.split('-')  # Convert to YYYY-MM-DD
        return f"{year}-{month}-{day}_{time_part}"  # Return formatted date & time
    except Exception as e:
        print(f"Error extracting date/time from filename {filename}: {e}")
        return "unknown_datetime"

# Determine well identifier
def get_well_identifier(row_start, row_end, column_start, column_end, reps, z, snake):
    row_indices = [chr(i) for i in range(ord(row_start.upper()), ord(row_end.upper()) + 1)]
    
    # Ensure columns are ordered correctly
    if column_start < column_end:
        columns = list(range(column_start, column_end + 1))  # Ascending order
    else:
        columns = list(range(column_start, column_end - 1, -1))  # Descending order

    wells_per_row = len(columns) * reps  # Total wells per row
    total_wells = len(row_indices) * wells_per_row  # Total number of wells

    if z >= total_wells:
        return None  # If beyond the expected range

    well_index = z // reps  # Determine well index
    current_row = row_indices[well_index // len(columns)]  # Row selection
    column_offset = well_index % len(columns)  # Get the specific column in this row

    # Handle snake pattern
    row_index = well_index // len(columns)
    if snake:
        if row_index % 2 == 0:
            current_column = columns[column_offset]  # Normal direction
        else:
            current_column = columns[-(column_offset + 1)]  # Reverse direction
    else:
        current_column = columns[column_offset]  # Regular linear order

    return f"{current_row}-{current_column}"

# Process FLIM data and save as TIFF
def process_flim_data(writepath, row_start, row_end, column_start, column_end, FOV_p_well, snake, NDD, binning=10):
    
    
    if NDD==0:
        xpix =80 
        ypix = 120
    if NDD==1:
        xpix =240 
        ypix = 240

    
    datafolder = create_folder_with_suffix(writepath)
    tcspcfolder = create_subfolder(datafolder, 'tcspc_stacks/')
    initialise_paths(tcspcfolder)

    pix, tbins, bin_size = 240, 1260, 10
    new_tbinsize = bin_size * binning
    file_num = len(os.listdir(directory))
    fullstack = np.zeros((pix, pix, tbins, file_num), dtype=np.uint16)
    intstack = np.zeros((pix,pix, file_num, 1), dtype=np.uint16) #mean arrival time for whole stack

    save_FLIM_ome_tiff(np.zeros((pix, pix, tbins // binning), dtype=np.uint16),
                  tcspcfolder + 'test.tif', 0, new_tbinsize, 'a')

    
    

    for z, file in enumerate(os.listdir(directory)):
        if z >= file_num:
            break
        filename = os.fsdecode(file)

        # Extract date from filename
        
        file_date = extract_datetime_from_filename(filename)
        # Get well identifier
        well_id = get_well_identifier(row_start, row_end, column_start, column_end, FOV_p_well, z, snake)
        if well_id is None:
            print("Reached the end of the well plate layout.")
            break

        # Generate FOV number
        fov_number = f"FOV{str(z+1).zfill(5)}"

        # Format new filename
        formatted_filename = f"{well_id} - {fov_number}_{file_date}.tiff"
        filepath = os.path.join(tcspcfolder, formatted_filename)

        #print(f"Saving as: {formatted_filename}")

        # Read PTU file
        ptu_file = PTUreader(read_path + filename, print_header_data=False)

        # Process data
        flim_data_stack, intensity_image = ptu_file.get_flim_data_stack(xpix, ypix, tbins, 7)
        
        
        if NDD==0:
            fullstack[:,:,:,z]= stitch_im_new1(flim_data_stack)
            intstack[:,:,z, :]= stitch_im_new1(np.expand_dims(intensity_image, axis=-1))
        else:
            fullstack[:,:,:,z]= (flim_data_stack[:,:,6,:])
            intstack[:,:,z, :]= (np.expand_dims(intensity_image[:,:,6], axis=-1))
        
        
        
        
        
        fullstack_binned = np.sum(fullstack[:, :, :, z].reshape(fullstack.shape[0:2] + (tbins // binning, binning)), axis=3).astype(np.uint16)
        save_FLIM_ome_tiff(fullstack_binned, filepath, 1, binning * 10, tcspcfolder)
        
        
        
        print(f"{(100 * (z+1) / file_num):.2f}%")

    intstack = intstack.squeeze()  
    ints = np.transpose(intstack, (2, 1,0))
    metadata={
         'axes': 'ZYX',
     }    
    with tifffile.TiffWriter(datafolder+'/intensitystack.tif', ome=True) as tif:
        tif.write(ints, metadata=metadata)   
    
    

    print("finished")

def parse_arguments():
    parser = argparse.ArgumentParser(description='TTTR processing.')
    parser.add_argument('--folder', type=str, required=True, help='Folder name for processed data.')
    parser.add_argument('--row_start', type=str, required=True, help='Starting row letter (A-H).')
    parser.add_argument('--row_end', type=str, required=True, help='Ending row letter (A-H).')
    parser.add_argument('--column_start', type=int, required=True, help='Starting column number (1-12).')
    parser.add_argument('--column_end', type=int, required=True, help='Ending column number (1-12).')
    parser.add_argument('--FOV_p_well', type=int, required=True, help='Number of fields of view (FOV) per well.')
    parser.add_argument('--snake', type=int, required=True, choices=[0, 1],
                        help='Set to 1 for snake pattern, 0 for standard pattern.')
    parser.add_argument('--NDD', type=int, required=True, choices=[0,1], help='1 for NDD acq.')
    
    return parser.parse_args()
# Main function
if __name__ == "__main__":
    args = parse_arguments()
    process_flim_data(
        writepath=f"D:/m3mbill/m3mdata/{args.folder}/",
        row_start=args.row_start,
        row_end=args.row_end,
        column_start=args.column_start,
        column_end=args.column_end,
        FOV_p_well=args.FOV_p_well,
        snake=args.snake,
        NDD = args.NDD
    )
    print("finished")
