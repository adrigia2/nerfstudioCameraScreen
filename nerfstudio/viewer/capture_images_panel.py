from typing import List
import torch
from viser import ViserServer
import viser
from PIL import Image
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.models.base_model import Model
from nerfstudio.viewer.render_state_machine import RenderStateMachine
from nerfstudio.viewer.utils import CameraState
from nerfstudio.viewer.viewer_elements import ViewerButton, ViewerCheckbox, ViewerDropdown, ViewerNumber, ViewerSlider, ViewerText, ViewerVec3

import matplotlib.pyplot as plt
import numpy as np

class CaptureImagesPanel:

    def __init__(
            self,
            server : ViserServer,
    ):
        self.server = server

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


        self.cameras = self.generate_multiple_camera_states()

    def set_render_state_machine(self, render_state_machine: RenderStateMachine) -> None:
        self.render_state_machine = render_state_machine

    def set_client(self, client: viser.ClientHandle) -> None:
        self.client = client

    def handle_btn(self) -> None:
        print("Button clicked!")
        for camera in self.cameras:
            print(f"Rendering image for camera at position {camera.c2w[:, 3].T}")
            image=self.render_state_machine._render_img(camera)["rgb"].cpu().numpy()
            # save image as png
            image_array = image  # È già in formato numpy grazie a .cpu().numpy()

            # Converti l'array numpy in un'immagine PIL
            image_pil = Image.fromarray((image_array * 255).astype(np.uint8))  # Scala a [0, 255] se l'array è in [0, 1]
            image_pil.save("output.png")

            #read the output.png file and send it to the client
            with open("output.png", "rb") as f:
                image_bytes = f.read()
                self.client.send_file_download("output.png", image_bytes)

            
    # Utility function to create camera state
    def create_camera_state(self, fov: float, aspect: float, position: str, camera_type: CameraType) -> CameraState:
        """Creates a CameraState for a given position."""
        # Default translation for the camera to be placed 1 unit away from the origin
        translation = {
            'front': torch.tensor([[0, 0, 1]]).T,
            'back': torch.tensor([[0, 0, -1]]).T,
            'up': torch.tensor([[0, 1, 0]]).T,
            'down': torch.tensor([[0, -1, 0]]).T
        }[position]

        # Define rotation matrices
        if position == "front":
            rotation = torch.eye(3)
        elif position == "back":
            rotation = torch.diag(torch.tensor([-1, -1, 1]))  # Flip x and y axes
        elif position == "up":
            rotation = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        elif position == "down":
            rotation = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        else:
            raise ValueError(f"Invalid position: {position}")

        # Combine rotation and translation to form c2w matrix
        c2w = torch.cat((rotation, translation), dim=1)  # Shape (3, 4)

        # Create the CameraState instance
        return CameraState(fov=fov, aspect=aspect, c2w=c2w, camera_type=
                CameraType.PERSPECTIVE
                if camera_type == "Perspective"
                else CameraType.FISHEYE
                if camera_type == "Fisheye"
                else CameraType.EQUIRECTANGULAR
                if camera_type == "Equirectangular"
                else CameraType.PERSPECTIVE)


    def generate_multiple_camera_states(self) -> List[CameraState]:
        # Example usage
        fov = 90.0  # Field of view in degrees
        aspect = 16/9  # Aspect ratio
        camera_type = CameraType.PERSPECTIVE

        # Generate CameraStates for different positions
        front_camera = self.create_camera_state(fov, aspect, "front", camera_type)
        back_camera = self.create_camera_state(fov, aspect, "back", camera_type)
        up_camera = self.create_camera_state(fov, aspect, "up", camera_type)
        down_camera = self.create_camera_state(fov, aspect, "down", camera_type)

        return [front_camera, back_camera, up_camera, down_camera]


    
