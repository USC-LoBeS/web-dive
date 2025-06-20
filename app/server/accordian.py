from shiny import ui
from ..utils import FILE_TYPE_COLORS, format_file_size

from shiny_cp import input_color_picker
from ..ui.module.input_file import file_input_ui


def create_accordian_name(idx, file):
    # Define filename
    ext = file["type"].lower()
    color_scheme = FILE_TYPE_COLORS.get(ext, FILE_TYPE_COLORS["default"])
    
    file_name = ui.div(
        ui.div(
            file["name"],
            class_="file-title",
            title=file["name"] 
        ),
        ui.div(
            ext.upper(),
            class_="file-ext",
            style=f"background-color: {color_scheme['bg']}; color: {color_scheme['text']}; display: none"
        ),
        class_="file-name"
    )

    return ui.div(
        file_name,
        class_="accordian-name"
    )

def create_file_card(idx, file):
    color_picker = ui.div(
        ui.p("Color:"),
        ui.span(
            input_color_picker(f"color_picker_{idx}"),
            style="z-index:100"
        ),
        class_="color-picker"
    )

    visibility_switch = ui.div(
        ui.p("Visible:"),
        ui.div(
        ui.input_switch(f"visible_switch_{idx}", "", file["visible"]),
        ),
        class_="visibility-switch"
    )

    stats_file = ui.div(
        ui.p("Stats file:"),
        ui.div(
            file_input_ui(f"stats_file_{idx}", ns=f"stats_file_{idx}", accept=[".csv"], multiple=False),
            # class_="custom-file-input"
        ),
        class_="stats-file"
    )

    meta_switch = ui.div(
        ui.div(
            ui.tooltip(
                ui.span("Method:  ",
                        ui.HTML('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-question-circle-fill mb-1" viewBox="0 0 16 16"><path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM5.496 6.033h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286a.237.237 0 0 0 .241.247zm2.325 6.443c.61 0 1.029-.394 1.029-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94 0 .533.425.927 1.01.927z"/></svg>')
                ),
                "Something here...",
                placement="right",
                id=f"tooltip_meta_{idx}"
            ),
        ),
        ui.div(
            ui.p("Center"),
            ui.span(
                ui.input_switch(f"meta_switch_{idx}", ""),
            ),
            ui.p("MeTA"),

            class_="meta-toggle"
        ),
        class_="meta-switch"
    )
    

    card_content = ui.div(
        color_picker,
        visibility_switch,

        stats_file,
        meta_switch,
        class_="file-card"
    )
    return card_content