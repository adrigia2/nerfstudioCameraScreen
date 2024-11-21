from typing import List
import torch

from PIL import Image
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.models.base_model import Model
from nerfstudio.viewer.render_state_machine import RenderStateMachine
from nerfstudio.viewer.utils import CameraState
from nerfstudio.viewer.viewer_elements import ViewerButton, ViewerCheckbox, ViewerDropdown, ViewerNumber, ViewerSlider, ViewerText, ViewerVec3

import matplotlib.pyplot as plt
import numpy as np

from nerfstudio.viewer.renderer_manager import RendererManager
import viser
from viser import ViserServer

class CaptureImagesPanel:

    def __init__(
            self,
            server : ViserServer,
    ):
        self.server = server
        self.renderer_manager = RendererManager()

        self.a = ViewerButton(name="My Button", cb_hook= lambda han: self.handle_btn())
        self.b = ViewerNumber(name="Number", default_value=1.0)
        self.c = ViewerCheckbox(name="Checkbox", default_value=False)
        self.d = ViewerDropdown(name="Dropdown", default_value="A", options=["A", "B"])
        self.e = ViewerSlider(name="Slider", default_value=0.5, min_value=0.0, max_value=1.0)
        self.f = ViewerText(name="Text", default_value="Hello World")
        self.g = ViewerVec3(name="3D Vector", default_value=(0.1, 0.7, 0.1))

        with self.server.gui.add_folder("TestOptions"):
            self.a.install(self.server)
            self.b.install(self.server)
            self.c.install(self.server)
            self.d.install(self.server)
            self.e.install(self.server)
            self.f.install(self.server)
            self.g.install(self.server)


        self.cameras = self.renderer_manager.generate_multiple_camera_states()

    def set_render_state_machine(self, render_state_machine: RenderStateMachine) -> None:
        self.renderer_manager.set_render_state_machine(render_state_machine)

    def set_client(self, client: viser.ClientHandle) -> None:
        self.client = client

    def handle_btn(self) -> None:
        print("Button clicked!")
        for camera in self.cameras:
            print(f"Rendering image for camera at position {camera.c2w[:, 3].T}")
            

            # save image as png
            image_array = self.renderer_manager.render_image(camera)  # È già in formato numpy grazie a .cpu().numpy()

            # Converti l'array numpy in un'immagine PIL
            image_pil = Image.fromarray((image_array * 255).astype(np.uint8))  # Scala a [0, 255] se l'array è in [0, 1]
            image_pil.save("output.png")

            #read the output.png file and send it to the client
            with open("output.png", "rb") as f:
                image_bytes = f.read()
                self.client.send_file_download("output.png", image_bytes)

            
    
