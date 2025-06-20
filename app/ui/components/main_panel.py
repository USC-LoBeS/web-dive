from shiny import ui

from .plot3d import plot3d
from .plot2d import plot2d
def main_panel_content():
    return [
        ui.navset_card_underline(
            ui.nav_panel(
                "3D",
                *plot3d()
            ),

            ui.nav_panel(
                "2D",
                # *plot2d()
                plot2d()
            ),

            ui.nav_panel(
                "Table",
                ui.output_ui("table_tabs")
            )
        )
    ]
