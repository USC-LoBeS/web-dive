import collections
from math import isnan
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from dipy.tracking.streamline import Streamlines, transform_streamlines
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .tract_segmentations import get_len, assignment_map_
from .constant import RGBA_TO_CMAP
import nibabel as nib

path = Path(__file__).parents[1] / "assets" / "brain_image.nii.gz"
affine_mask = nib.load(path).affine
mask_shape = nib.load(path).shape

def orient2rgba(v, alpha=1.0):
    """
    Converts orientation vector to RGBA color.

    Args:
        v (ndarray): Array of shape (N, 3) or (3,) representing orientation vectors.
        alpha (float): The alpha value to be appended to the color (default: 1.0).

    Returns:
        ndarray: RGBA values of shape (N, 4) or (4,).

    """
    if v.ndim == 1:
        r = np.linalg.norm(v)
        orient = np.abs(np.divide(v, r, where=r != 0))

    elif v.ndim == 2:
        orientn = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
        orientn.shape = orientn.shape + (1,)
        orient = np.abs(np.divide(v, orientn, where=orientn != 0))
    else:
        raise IOError(
            "Wrong vector dimension, It should be an array" " with a shape (N, 3)"
        )

    # Append an alpha channel to RGB value
    if v.ndim == 1:
        rgba = np.concatenate([orient, [alpha]])
    elif v.ndim == 2:
        alpha_channel = np.full((orient.shape[0], 1), alpha)
        rgba = np.concatenate([orient, alpha_channel], axis=1)

    return rgba

# Function to calculate streamline direction and assign color based on orientation
def change_color_based_on_flow(bundle):
    """
    Changes the color of each streamline based on its overall direction (e.g., first and last points).
    
    Args:
        bundle (list of ndarray): List of streamlines, where each streamline is an ndarray of shape (n_points, 3).
        colormap (function): A colormap function that takes an orientation (direction vector) and returns an RGB color.
    
    Returns:
        list: A list of RGB colors, one for each streamline.
    """
    streamline_colors = []

    for streamline in bundle:
        diff = np.diff(streamline, axis=0)
        diff = np.concatenate([diff[:1], diff], axis=0)
        orientations = np.asarray(diff)
        colors_tract = orient2rgba(orientations)
        streamline_colors.append(colors_tract)
    
    return streamline_colors

def get_colors(file):
    # If color set to transparent
    color_aggerate = sum(file["color"].values())
    if file["stats"]:
        csv_stats_caller =  Colors_csv(file["stats"])
        #streamline_colors = []
        color_key = ",".join(str(file["color"][k]) for k in ["r", "g", "b", "a"])
        color_map_ = RGBA_TO_CMAP.get(color_key, "viridis")
        colors = csv_stats_caller.assign_colors(map=color_map_)
        nb_streams = get_len(file["stats"])
        
        if file["meta"]==False:
            #print("Printing All the keyssss",file.keys())
            indx = assignment_map_(file["data"], file["data"],nb_streams,method="Center")
            #print(indx)
            point_idx = 0
            streamline_colors = [[] for _ in range(len(file["data"]))]
            for streamline_idx in range(len(file["data"])):
                num_points = len(file["data"][streamline_idx])
                # Get the color indices for this streamline's points
                streamline_indices = indx[point_idx:point_idx + num_points]
                # Map indices to colors and store as tuples
                streamline_colors[streamline_idx] = [tuple(colors[idx]) for idx in streamline_indices]
                point_idx += num_points
            #print("indx")
            return streamline_colors
        elif file["meta"]==True:
            # print("In Meta")
            if (np.array_equal(file["affine"], np.eye(4))): aff = affine_mask
            else: aff = file["affine"]
            indx = assignment_map_(file["file"], file["file"],nb_streams,method="Meta",bundle_shape=mask_shape,affine=aff)
            streamline_colors = [[] for _ in range(len(file["data"]))]
            point_idx = 0
            for streamline_idx in range(len(file["data"])):
                num_points = len(file["data"][streamline_idx])
                # Get the indices for this streamline and map them directly to colors
                streamline_colors[streamline_idx] = [
                    tuple([0.0, 0.0, 0.0]) if idx == 0 else tuple(colors[idx-1])
                    for idx in indx[point_idx:point_idx + num_points]
                ]
                point_idx += num_points

            return streamline_colors
    
    elif color_aggerate == 0:
        return change_color_based_on_flow(file["data"])

    else:
        color = file["color"]
        r, g, b, a = color['r'], color['g'], color['b'], color['a']
        rgba_color =  f'rgba({r},{g},{b},{a})'
        
        colors = [rgba_color] * len(file["data"])
        return colors
    
def load_brain_slice_image(idx = None, axis="axial"):
    file_path =Path(__file__).parents[1] / "assets" / "brain_image.nii.gz"
    img = nib.load(file_path)
    data = img.get_fdata()

    if axis == 'axial':
        mid_index = idx if idx else data.shape[2] // 2
        slice_2d = data[:, :, mid_index]
    elif axis == 'coronal':
        mid_index = idx if idx else data.shape[1] // 2
        slice_2d = data[:, mid_index, :]
    elif axis == 'sagittal':
        mid_index = idx if idx else data.shape[0] // 2
        slice_2d = data[mid_index, :, :]
    else:
        raise ValueError("Invalid axis. Choose 'axial', 'coronal', or 'sagittal'.")
    
    return slice_2d

def align_streamlines_to_mni(data):
    transformed_data = []
    for file_data in data:
        streamlines = file_data["data"]
        # pts = [item for sublist in streamlines for item in sublist]
        
        # transformed_streamlines = nib.affines.apply_affine(np.linalg.inv(affine_mask), pts)
        transformed_streamlines = []
        # Inverse of the affine matrix
        inv_affine = np.linalg.inv(affine_mask)
        # Process each streamline separately
        for streamline in streamlines:
            # Convert streamline to numpy array if it isn't already
            streamline_array = np.asarray(streamline)
            # Apply transformation to all points in the current streamline
            transformed_points = nib.affines.apply_affine(inv_affine, streamline_array)
            # Convert back to list and append to results
            transformed_streamlines.append(transformed_points)
        transformed_streamlines = np.array(transformed_streamlines,dtype=object)
        transformed_streamlines = Streamlines(transformed_streamlines)
        transformed_data.append({
            "id": file_data["id"],
            "name": file_data["name"],
            "data": transformed_streamlines,
            "size": file_data["size"],
            "type": file_data["type"],
            "visible": file_data["visible"],
            "color": file_data["color"],
            "stats": file_data["stats"],
            "meta": file_data["meta"],
            "dim": file_data["dim"],
            "affine": affine_mask,
            "file": file_data["file"]
        })
    
    return transformed_data

# def align_streamlines_to_mni(data):
#     transformed_data = []
    
#     for file_data in data:
#         streamlines = file_data["data"]
        
#         # Apply DIPY's transform_streamlines function
#         transformed_streamlines = transform_streamlines(streamlines, affine_mask)
        
#         # Wrap transformed streamlines as a DIPY Streamlines object
#         transformed_streamlines = Streamlines(transformed_streamlines)
        
#         # Store transformed data
#         transformed_data.append({
#             "id": file_data["id"],
#             "name": file_data["name"],
#             "data": transformed_streamlines,
#             "size": file_data["size"],
#             "type": file_data["type"],
#             "visible": file_data["visible"],
#             "color": file_data["color"],
#             "stats": file_data["stats"],
#             "meta": file_data["meta"],
#             "dim": file_data["dim"],
#             "affine": affine_mask,
#             "file": file_data["file"]
#         })
    
#     return transformed_data

def format_file_size(size_in_bytes):
    """
    Convert file size in bytes to a human-readable format.
    
    Parameters:
    - size_in_bytes (int): The file size in bytes.
    
    Returns:
    - str: Human-readable file size.
    """
    if size_in_bytes == 0:
        return "0 Bytes"
    
    size_units = ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    index = 0

    while size_in_bytes >= 1024 and index < len(size_units) - 1:
        size_in_bytes /= 1024.0
        index += 1

    return f"{size_in_bytes:.2f} {size_units[index]}"

class Colors_csv():
    def __init__(self,stats_csv=None):
        self.csv_flag = None
        self.colors_from_csv = []
        self.stats_csv = stats_csv
        self.map = None
    def load_csv(self): 
        self.df = pd.read_csv(self.stats_csv)
    def intialize(self,log_p_value=False):
        if log_p_value:
            self.df['P_value'] = -np.log10(self.df['P_value'])
            self.col_name = 'P_value'
        else:
            self.col_name = 'Value'
            
        self.min_value = self.df[self.col_name].min()       
        self.max_value = self.df[self.col_name].max()  

        
    def assign_colors(self,map="viridis",range_value=[],log_p_value=False,threshold=None,output=None,filename='_color_bar.pdf',group=0):
        self.map = map
        if group==0: self.load_csv()
        self.intialize(log_p_value)
        cmap = matplotlib.cm.get_cmap(map)
        
        di = {}
        if range_value and len(range_value)>0:
            self.min_value = range_value[0]
            self.max_value = range_value[1]
            self.csv_flag = True
            self.df.sort_values(by=['Labels'],inplace=True)

        norm = mcolors.Normalize(vmin=self.min_value, vmax=self.max_value)
        self.df['Value_n'] = norm(self.df[self.col_name])
        
        for i,k,j in zip(self.df['Value_n'],self.df['P_value'],self.df['Labels']):
            if self.csv_flag == True:
                if k>threshold:
                    r=g=b=0.5
                else:
                    if i!= None and self.min_value<=i<=self.max_value:  
                        r,g,b,a = cmap(i)
                    if i>self.max_value:
                        r,g,b,a = cmap(self.max_value)
                    if i<self.min_value:
                        r,g,b,a = cmap(self.min_value)
            else:
                r,g,b,a = cmap(i)
            di[j] = [r,g,b]
        clean_dict = {k: di[k] for k in di if not isnan(k)}
        clean_dict = collections.OrderedDict(sorted(clean_dict.items()))
        for k, v in clean_dict.items():
            self.colors_from_csv.append(v)
        self.colors_from_csv = np.asarray(self.colors_from_csv)

        ## Save the color bar images
        
        # fig, ax = plt.subplots(figsize=(6, 1))

        # print(self.df[self.col_name])
        if output!=None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=ax,orientation='horizontal')
            # cb.ax.tick_params(labelsize=10) 
            plt.savefig((output + filename) , bbox_inches='tight', dpi=300)
        # print(self.colors_from_csv)
        return self.colors_from_csv
    
    def grouped_colors(self):
        self.load_csv()
        grouped = self.df.groupby('Name')

        for name, group in grouped:
            self.df = grouped
            return self.assign_colors(self,map,range_value=[],log_p_value=False,threshold=None,output=None,group=1)