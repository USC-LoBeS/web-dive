from shiny import ui
from .components.sidebar import sidebar_content
from .components.main_panel import main_panel_content

from pathlib import Path

css_file = Path(__file__).parent / "styles.css"

def create_app_layout():
   return ui.page_fluid(
      ui.head_content(
         ui.include_css(css_file),
         ui.tags.link(rel="icon", type="image/x-icon", href="logo_bg.ico")
      ),
      ui.panel_title(
            ui.div(
              ui.div(
                 ui.div(
                    ui.a(
                     ui.img(src="Logo.svg", alt="Logo"),
                       href = "https://github.com/USC-LoBeS/dive",
                       target="_blank"
                    ),
                    class_="logo-container"
                 ),
                 ui.div(
                    ui.h2("Diffusion Visualization Explorer"),
                    class_="logo-name-container"
                 ),
                 class_="name-container"
              ),
              class_="topbar-container"
            )
      ),
      ui.layout_sidebar(
         ui.sidebar(
            width=400,
            *sidebar_content()
         ),
         *main_panel_content(),
         class_="full-page"
      ),      
   )
