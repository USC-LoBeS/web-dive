from shiny import ui, module, reactive, Inputs, Outputs, Session

import asyncio
from pathlib import Path

@module.ui
def file_input_ui(ns = "", button_label="Browse...", accept = None, multiple = False):
    css_file = Path(__file__).parent / "styles.css"
    Session
    return ui.div(
        ui.include_css(css_file),
        ui.input_file(
            id = "uploader",
            label="",
            button_label=button_label,
            multiple=multiple,
            accept=accept
        ),
        ui.div(id=f"progress-box-{ns}",class_="custom-progress-box"),

        # add js to change behaviour
        ui.tags.script("""
            const spinner = document.createElement("div")
            spinner.classList.add("spinner")
            
            // Define a function to create a new checkmark SVG
            function createCheckmark() {
                const svgNS = 'http://www.w3.org/2000/svg';
                const checkmark = document.createElementNS(svgNS, 'svg');
                checkmark.setAttribute('class', 'checkmark');
                checkmark.setAttribute('viewBox', '0 0 52 52');

                const check = document.createElementNS(svgNS, 'path');
                check.setAttribute('class', 'checkmark__check');
                check.setAttribute('fill', 'none');
                check.setAttribute('d', 'M14.1 27.2l7.1 7.2 16.7-16.8');
                checkmark.appendChild(check);

                return checkmark;
            }
                      
            window.Shiny.addCustomMessageHandler('status', function(value){
                const statusIndicator = document.getElementById(`progress-box-${value.ns}`);
                if(value.msg === "uploading"){
                    statusIndicator.classList.remove("complete");
                    statusIndicator.classList.add("active");
                       
                    statusIndicator.appendChild(spinner)
                }
                else if(value.msg === "complete"){
                    statusIndicator.classList.remove("active");
                    statusIndicator.classList.add("complete");
                       
                    statusIndicator.removeChild(statusIndicator.firstElementChild)
                    const newCheckmark = createCheckmark()
                    statusIndicator.appendChild(newCheckmark)
                }
            });
        """),
        class_="custom-file-input-container"
    )

@module.server
def file_input_server(input: Inputs, output: Outputs, session: Session, _on_upload=None):

    @reactive.effect
    @reactive.event(input.uploader)
    async def _start_upload():
        files = input.uploader()
        await session.send_custom_message("status", {
                "msg": "uploading",
                "ns": session.ns
                })
        
        # Simulate upload progress
        for i in range(1, 11):
            await asyncio.sleep(0.5)
                
        if files is not None:
            await session.send_custom_message("status", {
                "msg": "complete",
                "ns": session.ns
                })

            if _on_upload is not None:
                _on_upload(files)

    # Return the reactive event
    if input.uploader is not None:
        return {"file_selected": input.uploader}