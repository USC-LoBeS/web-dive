from shiny import ui
from ..module.input_file import file_input_ui

def sidebar_content():
    """Returns the UI elements for the sidebar."""
    return [
        ui.div(
                ui.p("Upload a bundle: "),
                ui.div(
                file_input_ui("trk_file", ns="trk_file", accept=[".trk"], multiple = True),
                # ui.input_file("trk_file", "", accept=[".trk"], multiple=True),
                class_="custom-file-input"
                ),
                class_="main-file-input"
        ),
        ui.output_ui("file_accordian"),
    ]