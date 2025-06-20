from shiny import ui
import nibabel as nib

from pathlib import Path
from ...utils import VIEWS, slice_ranges

def plot2d():
    return ui.output_ui("plot_2d")
        # [
        # ui.input_select(
        #     id="orientation", 
        #     label="Select Orientation", 
        #     choices=list(slice_ranges.keys()), 
        #     selected="axial"
        # ),
        # ui.input_slider(
        #     id="slice_selector",
        #     label="",
        #     min=slice_ranges["axial"]["min"],
        #     max=slice_ranges["axial"]["max"],
        #     value=slice_ranges["axial"]["max"] // 2
        #     ),

    #     ui.row(
    #         *[ui.column(4, ui.card(
    #             ui.card_header(view.capitalize()),
    #             ui.output_ui(f"plot_{view}"),
    #             ui.input_slider(f"plot_{view}_slider",
    #                             "", # No lable
    #                             min=slice_ranges[view]["min"], 
    #                             max=slice_ranges[view]["max"], 
    #                             value=slice_ranges[view]["max"] // 2, # Load Mid slice by default
    #                         )
    #         )) for view in VIEWS]
    #     )
    # ] 
        