import numpy as np

import plotly.graph_objects as go
from joblib import Parallel, delayed

from ..utils import get_colors, load_brain_slice_image, align_streamlines_to_mni


# Global variable to store current camera state
_current_camera = None
#print(_current_camera)
def create_single_streamline_trace(streamline, color):
    """Previous implementation remains the same"""
    #print(len(streamline))
    x, y, z = streamline[:, 0], streamline[:, 1], streamline[:, 2]
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(color=color, width=2),
        showlegend=False,
    )
def create_streamline_traces(file):
    """Previous implementation remains the same"""
    #print("Doing GetCOLORSSSSS",file.keys())
    colors = get_colors(file)
    #print("Done GetCOLORSSSSS",file.keys())
    streamlines = file["data"]
    #print("loading strealinessssss")
    traces = Parallel(n_jobs=-1)(
        delayed(create_single_streamline_trace)(streamline, colors[idx])
        for idx, streamline in enumerate(streamlines)
    )
    #print("Loaded")
    return traces

def process_file(data):
    """Helper function for parallel processing"""
    return create_streamline_traces(data)

def get_default_views():
    """Return dictionary of default camera views"""
    return {
        'Sagittal': dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2.5, y=0, z=0)
        ),
        'Coronal': dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=2.5, z=0)
        ),
        'Axial': dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=2.5)
        )
    }

def plot_multiple_files(file_streamlines_data, preserve_camera=True, is_2d_view=False):
    """
    Create a Plotly figure with multiple streamlines and view control buttons.
    
    Parameters:
        file_streamlines_data: List of dictionaries containing streamline data
        preserve_camera: Boolean to indicate if camera position should be preserved
        is_2d_view: Boolean to indicate if 2D brain slices should be shown
        progress: A progress UI element to display progress bar
    """
    # Get default views based on whether we're in 2D or 3D mode
    if is_2d_view:
        # Use the 2D-specific camera settings
        views = {
            'Axial': dict(
                up=dict(x=0, y=-1, z=0), 
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0, z=2.5)
            ),
            'Coronal': dict(
                up=dict(x=0, y=0, z=1), 
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=2.5, z=0)
            ),
            'Sagittal': dict(
                up=dict(x=0, y=0, z=1), 
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-2.5, y=0, z=0)
            )
        }

        # Transform streamlines to mni 
        file_streamlines_data = align_streamlines_to_mni(file_streamlines_data)
    else:
        # Use the standard 3D views
        views = get_default_views()

    global _current_camera
    fig = go.Figure()
    
    # Process streamlines
    all_traces = Parallel(n_jobs=-1)(
        delayed(process_file)(data) for data in file_streamlines_data if bool(data["visible"])
    )
    all_traces = [trace for sublist in all_traces for trace in sublist]
    fig.add_traces(all_traces)
    
    # Store the number of streamline traces for visibility management
    n_streamline_traces = len(fig.data)

    
    # Define slice parameters for each view
    slice_parameters = {
        'Sagittal': {'axis': 'sagittal', 'slice_position': 127},
        'Coronal': {'axis': 'coronal', 'slice_position': 127},
        'Axial': {'axis': 'axial', 'slice_position': 127}
    }
    
    # For 2D view, pre-add all slices
    if is_2d_view:
        for view_name, params in slice_parameters.items():
            # Load the appropriate slice for this view
            slice_2d = load_brain_slice_image(params['slice_position'], params['axis'])
            
            # Add the slice to the figure
            add_slice_to_figure(
                fig, 
                slice_2d, 
                params['axis'], 
                params['slice_position']
            )
            
            # Make all slices invisible initially except for Sagittal (default view)
            if view_name != 'Axial':
                fig.data[-1].visible = False

    # Create buttons with icons
    buttons = []
    for view_name, camera in views.items():
        if is_2d_view and view_name in slice_parameters:
            visibility = [True] * n_streamline_traces  # All streamlines visible
            
            # Add visibility settings for the slice traces
            for i, slice_view in enumerate(slice_parameters.keys()):
                visibility.append(slice_view == view_name)
            
            button = dict(
                args=[
                    {'visible': visibility},    
                    {'scene.camera': camera}
                ],
                label=view_name,
                method='update'  
            )
        else:
            # Original behavior for 3D view or views without slices
            button = dict(
                args=[{'scene.camera': camera}],
                label=view_name,
                method='relayout'
            )
        buttons.append(button)

    # Use stored camera position if available and preservation is requested
    if preserve_camera and _current_camera:
        camera_settings = _current_camera
    else:
        # Set default camera based on whether we're in 2D or 3D mode
        if is_2d_view:
            camera_settings = views['Sagittal']  # Default for 2D
        else:
            camera_settings = get_default_views()['Sagittal']  # Default for 3D

    # Add button menu to layout
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                # showactive=False,
                buttons=buttons,
                x=0.1,
                xanchor='left',
                y=0.9,
                yanchor='top',
                direction='right',
                pad={"r": 10, "t": 10},
                bgcolor='rgba(255, 255, 255, 0.9)',
                font=dict(size=12)
            )
        ],
        scene=dict(
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
            zaxis=dict(showgrid=False, visible=False),
            camera=camera_settings,
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=800,
    )

    return fig

# def plot_multiple_files(file_streamlines_data, preserve_camera=True, is_2d_view = False):
#     """
#     Create a Plotly figure with multiple streamlines and view control buttons.
    
#     Parameters:
#         file_streamlines_data: List of dictionaries containing streamline data
#         preserve_camera: Boolean to indicate if camera position should be preserved
#     """
#     global _current_camera
#     fig = go.Figure()
#     #print(file_streamlines_data)
#     # Process streamlines
#     all_traces = Parallel(n_jobs=-1)(
#         delayed(process_file)(data) for data in file_streamlines_data if bool(data["visible"])
#     )
#     all_traces = [trace for sublist in all_traces for trace in sublist]
#     fig.add_traces(all_traces)

#     # Get default views
#     views = get_default_views()
#     slice_parameters = {
#         'Axial': {'axis': 'axial', 'slice_position': 127},
#         'Coronal': {'axis': 'coronal', 'slice_position': 127},
#         'Sagittal': {'axis': 'sagittal', 'slice_position': 127}
#     }
#     # Create buttons with icons
#     buttons = []
#     for view_name, camera in views.items():
#         if is_2d_view and view_name in slice_parameters:
#             slice_2d = load_brain_slice_image(None, axis=slice_parameters[view_name]["axis"])
#             button = dict(
#                 args=[
#                     {'scene.camera': camera},
#                     # This function will be executed when the button is clicked
#                     [add_slice_to_figure, [fig, slice_2d, slice_parameters[view_name]['axis'], 
#                                           slice_parameters[view_name]['slice_position']]]
#                 ],
#                 label=view_name,
#                 method='relayout'  # Multiple methods can be specified
#             )
                        
#         else:
#             button = dict(
#                 args=[{
#                     'scene.camera': camera
#                 }],
#                 label=view_name,
#                 method='relayout'
#             )
#         buttons.append(button)

#     # Use stored camera position if available and preservation is requested
#     camera_settings = _current_camera if preserve_camera and _current_camera else views['Sagittal']
#     # slice_2d = load_brain_slice_image(idx = None ,axis=view)
#     if is_2d_view:
#         default_view = 'Sagittal'  # Default view
#         slice_2d = load_brain_slice_image(None, axis= slice_parameters[default_view]["axis"])
#         add_slice_to_figure(
#             fig, 
#             slice_2d,
#             slice_parameters[default_view]['axis'], 
#             slice_parameters[default_view]['slice_position']
#         )

#     # Add button menu to layout
#     fig.update_layout(
#         updatemenus=[
#             dict(
#                 type='buttons',
#                 showactive=False,
#                 buttons=buttons,
#                 x=0.1,
#                 xanchor='left',
#                 y=0.9,
#                 yanchor='top',
#                 direction='right',
#                 pad={"r": 10, "t": 10},
#                 bgcolor='rgba(255, 255, 255, 0.9)',
#                 font=dict(size=12)
#             )
#         ],
#         scene=dict(
#             xaxis=dict(showgrid=False, visible=False),
#             yaxis=dict(showgrid=False, visible=False),
#             zaxis=dict(showgrid=False, visible=False),
#             camera=camera_settings,
#             aspectmode='data',
#         ),
#         margin=dict(l=0, r=0, t=0, b=0),
#         height=800,
#     )

#     return fig

def add_slice_to_figure(fig, slice_2d, axis='axial', slice_position=127):
    """
    Add a 2D brain slice to the 3D figure with correct orientation.
    
    Parameters:
        fig: Plotly figure to add the slice to
        slice_2d: 2D numpy array containing the slice data
        axis: Orientation of the slice ('axial', 'coronal', or 'sagittal')
        slice_position: Position of the slice along the specified axis
    """
    if slice_2d is None:
        # If slice_2d is not provided, load it
        slice_2d = load_brain_slice_image(slice_position, axis)
    
    x_dim, y_dim = slice_2d.shape
    
    # Create coordinate grids based on the dimensions of the slice
    x = np.linspace(0, x_dim - 1, x_dim)
    y = np.linspace(0, y_dim - 1, y_dim)
    x_grid, y_grid = np.meshgrid(x, y)
    
    if axis == 'axial': 
       
        slice_2d = np.rot90(slice_2d, k=1)  

        z_grid = np.full_like(x_grid, slice_position)

        fig.add_trace(go.Surface(
            z=z_grid, 
            x=x_grid, 
            y=y_grid,
            surfacecolor=slice_2d, 
            colorscale='Gray',
            showscale=False
        ))
        
    elif axis == 'coronal':
        slice_2d = np.rot90(slice_2d, k=3)

        y_grid_fixed = np.full_like(x_grid, slice_position)
        fig.add_trace(go.Surface(
            z=y_grid,  # y_grid becomes z_grid for coronal view
            x=x_grid,
            y=y_grid_fixed,
            surfacecolor=slice_2d, 
            colorscale='Gray',
            showscale=False
        ))
        
    elif axis == 'sagittal': 
        slice_2d = np.rot90(slice_2d, k=1)
        slice_2d = np.flipud(slice_2d)

        x_grid_fixed = np.full_like(y_grid, slice_position)

        fig.add_trace(go.Surface(
            z=y_grid,  # x_grid becomes z_grid for sagittal view
            x=x_grid_fixed,
            y=x_grid,
            surfacecolor=slice_2d, 
            colorscale='Gray',
            showscale=False
        ))

# def add_slice_to_figure(fig, slice_2d, axis='axial', slice_position=127):
#     x_dim, y_dim = slice_2d.shape
#     #print("Add Slice_to_fig",slice_2d.shape)
#     x = np.linspace(0, x_dim - 1, x_dim)
#     y = np.linspace(0, y_dim - 1, y_dim)
#     x, y = np.meshgrid(x, y)
    
#     if axis == 'axial':  # Axial (XY plane, constant Z)
#         slice_2d = np.rot90(slice_2d, k =3)
#         z = np.full_like(x, slice_position)
#         x,y = y,x
#         #print("HERE")
#     elif axis == 'coronal':  # Coronal (XZ plane, constant Y)
#         slice_2d = np.rot90(slice_2d, k =3)
#         z = y
#         y = np.full_like(x, slice_position)
#     elif axis == 'sagittal':  # Sagittal (YZ plane, constant X)
#         slice_2d = np.rot90(slice_2d, k =1)
#         z = x
#         y = y
#         x = np.full_like(z, slice_position)

#     fig.add_trace(go.Surface(
#         z=z, x=x, y=y,
#         surfacecolor=slice_2d, 
#         colorscale='Gray',
#         # opacity=0.5,
#         showscale=False
#     ))



def plot_2d_view(file_streamlines_data, view, idx=None):
    fig = go.Figure()

    # Align data to MNI space
    transformed_data = align_streamlines_to_mni(file_streamlines_data)
    
    #print("Data is loaded",transformed_data.keys)
    # Parallelize processing of streamlines from different files
    def process_file(data):
        if bool(data["visible"]):
            return create_streamline_traces(data)
        return None
    
    # Using joblib's Parallel to process multiple files at once
    all_traces = Parallel(n_jobs=-1)(
        delayed(process_file)(data) for data in transformed_data
    )
    #("All traces")
    # Flatten the list of traces (list of lists)
    all_traces = [trace for sublist in all_traces for trace in sublist]
    
    # Add all traces to the figure
    fig.add_traces(all_traces)
    
    slice_2d = load_brain_slice_image(idx ,axis=view)
    #print("Slice 2d loaded",slice_2d)
    # Simplified camera and layout settings
    camera_settings = {
        "axial": dict(up=dict(x=0, y=0, z=1), eye=dict(x=0, y=0, z=2.5)),
        "coronal": dict(up=dict(x=0, y=0, z=1), eye=dict(x=0, y=2.5, z=0)),
        "sagittal": dict(up=dict(x=1, y=0, z=0), eye=dict(x=2.5, y=0, z=0))
    }

    add_slice_to_figure(fig, slice_2d, axis=view, slice_position=idx)
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
            zaxis=dict(showgrid=False, visible=False),
            camera=dict(
                up=camera_settings[view]['up'],
                center=dict(x=0, y=0, z=0),
                eye=camera_settings[view]['eye']
            ),
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=800,
        uirevision=True  # This preserves UI state including camera position
    )

    return fig

def update_visualization(file_streamlines_data, fig=None):
    """
    Update the visualization while preserving camera angle.
    
    Parameters:
        file_streamlines_data: Updated streamline data
        fig: Existing figure to update (optional)
    """
    global _current_camera
    
    if fig is not None:
        # Store the current camera position
        _current_camera = fig.layout.scene.camera
    
    # Create new figure with preserved camera
    return plot_multiple_files(file_streamlines_data, preserve_camera=True)