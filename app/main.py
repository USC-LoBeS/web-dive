import os
from copy import deepcopy
from pathlib import Path

from shiny import App, render, ui, reactive

import nibabel as nib

import plotly.graph_objects as go
import pandas as pd

from .ui.layout import create_app_layout
from .ui.module.input_file import file_input_server
from .server.tracks import plot_multiple_files, plot_2d_view
from .server.accordian import create_accordian_name, create_file_card
from .utils import VIEWS, slice_ranges

def server(input, output, session):
    streamlines_store = reactive.Value(None)
    processed_files = reactive.Value(None)
    camera_state = reactive.Value(None)
    
    def update_store(files):
        streamlines_store.set(files)
        
    file_input_server("trk_file", _on_upload=update_store)

    # Load .tck file
    @reactive.Effect
    def _load_file():
        uploaded_files = streamlines_store.get()
        if uploaded_files and len(uploaded_files) > 0:
           
            try:
                data = []
                for idx, file in enumerate(uploaded_files):
                    uploaded_file_path = file["datapath"]
                    trk_data = nib.streamlines.load(uploaded_file_path)
                    ext = os.path.splitext(file["name"])[-1]
                    data.append({
                        "id": idx,
                        "name": file["name"],
                        "data": trk_data.streamlines,
                        "size": file["size"],
                        "type": ext,
                        "visible": True,
                        "color": {"r": 0, "g": 0, "b": 0, "a": 0},
                        "stats": None,
                        "meta": False,
                        "dim": trk_data.header["dimensions"],
                        "affine": trk_data.affine,
                        "file": trk_data
                    })
                
                processed_files.set(data)
            except Exception as e:
                processed_files.set(None)
                print(f"Error loading file: {e}")
            

    # Output plotly figure
    @output 
    @render.ui
    def tract_plot():
        files = processed_files.get()
        if files is None or len(files) <= 0:
            fig = go.Figure()
            plot = fig.add_annotation(
                text="Please Upload a .trk file",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return ui.HTML(plot.to_html(include_plotlyjs="cdn", full_html=False))
        else:
            try:
                # Get figure for each file
                current_camera_state = camera_state.get() or {
                    'up': {'x': 0, 'y': 0, 'z': 1},
                    'center': {'x': 0, 'y': 0, 'z': 0},
                    'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}

                }
                progress = ui.Progress(min=0, max=100)
                progress.set(message="Processing Files", detail="This may take a while...")

                # Init progress
                progress.set(value=10, message="Processing Files", detail="This may take a while...")
                with reactive.isolate():
                    plot = plot_multiple_files(files)
                    progress.set(value=100, message="Processing Files", detail="This may take a while...")
                    progress.close()

                return ui.HTML(plot.to_html(include_plotlyjs="cdn", full_html=False))
            except Exception as e:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Error displaying streamlines: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))

    @output
    @render.ui
    @reactive.event(streamlines_store)
    def file_accordian():
        uploaded_files = streamlines_store.get()
        if not uploaded_files or len(uploaded_files) == 0:
            return ui.HTML("")
        
        files = processed_files.get()
        if not files:
            return ui.HTML("No valid streamlines data available.")
        
        accordian_items = []
        for idx, file in enumerate(files):
            panel = ui.accordion_panel(
                create_accordian_name(idx, file),
                create_file_card(idx, file),
                value=f"file_accordian_panel_{idx}",
                open = idx == 0,
                style="padding: 0.5rem 1rem;",
                class_="accordian-panel"
            )
            accordian_items.append(panel)

            # Track Effects
            create_color_effect(idx)
            create_visibility_effect(idx)

            # Stats Effects
            create_method_effect(idx)
            create_stats_file_effect(idx)
        
        return ui.accordion(*accordian_items)
    
    @output
    @render.ui
    def plot_2d():
        files = processed_files.get()

        if files is None or len(files) < 1:
            return ui.p("Please Upload a file")
        

        plot = plot_multiple_files(files, is_2d_view=True)
        return ui.HTML(plot.to_html(include_plotlyjs="cdn", full_html=False))
        

    @output
    @render.ui
    def table_tabs():
        files = processed_files.get()

        if not files or len(files) <= 0:
            return "Please Upload a file"
        tabs = []
        for idx, file in enumerate(files):
            if not file['stats']:
                tab = ui.nav_panel(file["name"], ui.p("No Stats file provided for the bundle", class_="p-3"))
            else:
                df = pd.read_csv(file['stats'])
                tab = ui.nav_panel(file["name"], ui.output_data_frame(f"table_output_{idx}"))
                populate_table_tab(idx, df)
            tabs.append(tab)
        
        return ui.navset_tab(*tabs)

    def populate_table_tab(idx, df):
        @output(id=f"table_output_{idx}")
        @render.data_frame
        def _():
            return render.DataGrid(
                df,
                width="100%",
                height=400
            )
            
    @reactive.Effect
    def update_slider():
        selected_orientation = input.orientation()
        min_val = slice_ranges[selected_orientation]["min"]
        max_val = slice_ranges[selected_orientation]["max"]
        
        session.send_input_message("slice_selector",{
            "min":min_val,
            "max":max_val,
            "value": max_val // 2
        })


    # UI EFFECTS FROM SIDEBAR
    def create_color_effect(idx):
        @reactive.Effect
        @reactive.event(input[f"color_picker_{idx}"])
        def update_file_color():

            files = processed_files.get()
            updated_files = deepcopy(files)

            picker_id = f"color_picker_{idx}"
            new_color = input[picker_id].get()

            # Check if the color is actually different
            # color_key = ",".join(str(updated_files[idx]["color"][k]) for k in ["r", "g", "b", "a"])
            if updated_files[idx]["color"] != new_color:
                updated_files[idx]["color"] = new_color
                processed_files.set(updated_files)
    
    def create_visibility_effect(idx):
        @reactive.Effect
        @reactive.event(input[f"visible_switch_{idx}"])
        def update_file_visibility():
            files = processed_files.get()
            updated_files = deepcopy(files)

            switch_id = f"visible_switch_{idx}"
            new_visibility = input[switch_id].get()

            if updated_files[idx]["visible"] != new_visibility:
                updated_files[idx]["visible"] = new_visibility
                processed_files.set(updated_files)

    def create_method_effect(idx):
        @reactive.Effect
        @reactive.event(input[f"meta_switch_{idx}"])
        def update_segmentation_method():
            files = processed_files.get()
            updated_files = deepcopy(files)

            switch_data = bool(input[f"meta_switch_{idx}"].get())

            if updated_files[idx]["meta"] != switch_data:
                updated_files[idx]["meta"] = switch_data
                processed_files.set(updated_files)
               
    def create_stats_file_effect(idx):

        file_input = file_input_server(f"stats_file_{idx}")
        @reactive.Effect
        @reactive.event(file_input["file_selected"])
        def update_stats_file():
            files = processed_files.get()
            update_files = deepcopy(files)

            datapath = file_input[f"file_selected"].get()[0]["datapath"]
            
            if datapath and update_files[idx]["stats"] != datapath:
                update_files[idx]["stats"] = datapath
                processed_files.set(update_files)

app_ui = create_app_layout()
static_folder = Path(__file__).parent / "assets"
app = App(app_ui, server, static_assets=static_folder)

if __name__ == "__main__":
    app.run()