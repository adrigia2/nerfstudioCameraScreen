from typing import List
import torch
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.viewer.render_state_machine import RenderStateMachine
from nerfstudio.viewer.utils import CameraState

class RendererManager:
    def __init__(self, render_state_machine: RenderStateMachine = None):
        self.render_state_machine = render_state_machine

    def set_render_state_machine(
        self, render_state_machine: RenderStateMachine
    ) -> None:
        self.render_state_machine = render_state_machine

    # Utility function to create camera state
    def create_camera_state(
        self, fov: float, aspect: float, position: str, camera_type: CameraType
    ) -> CameraState:
        """Creates a CameraState for a given position."""
        # Default translation for the camera to be placed 1 unit away from the origin
        translation = {
            "front": torch.tensor([[0, 0, 1]]).transpose(0, 1),
            "back": torch.tensor([[0, 0, -1]]).transpose(0, 1),
            "up": torch.tensor([[0, 1, 0]]).transpose(0, 1),
            "down": torch.tensor([[0, -1, 0]]).transpose(0, 1),
        }[position]

        # Define rotation matrices
        if position == "front":
            # 1 0 0
            # 0 1 0
            # 0 0 1
            rotation = torch.eye(3)
        elif position == "back":
            # rotate 180 degrees around y axis
            # -1 0 0
            # 0 1 0
            # 0 0 -1
            rotation = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        elif position == "up":
            # rotate 90 degrees around x axis
            # 1 0 0
            # 0 0 1
            # 0 -1 0
            rotation = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        else:
            raise ValueError(f"Invalid position: {position}")

        # Combine rotation and translation to form c2w matrix
        c2w = torch.cat((rotation, translation), dim=1)  # Shape (3, 4)

        # Create the CameraState instance
        return CameraState(
            fov=fov,
            aspect=aspect,
            c2w=c2w,
            camera_type=(
                CameraType.PERSPECTIVE
                if camera_type == "Perspective"
                else (
                    CameraType.FISHEYE
                    if camera_type == "Fisheye"
                    else (
                        CameraType.EQUIRECTANGULAR
                        if camera_type == "Equirectangular"
                        else CameraType.PERSPECTIVE
                    )
                )
            ),
        )

    def generate_multiple_camera_states(self) -> List[CameraState]:
        # Example usage
        fov = 70.0  # Field of view in degrees
        aspect = 16 / 9  # Aspect ratio
        camera_type = CameraType.PERSPECTIVE

        # Generate CameraStates for different positions
        front_camera = self.create_camera_state(fov, aspect, "front", camera_type)
        back_camera = self.create_camera_state(fov, aspect, "back", camera_type)
        up_camera = self.create_camera_state(fov, aspect, "up", camera_type)

        return [front_camera, back_camera, up_camera]

    def render_image(self, camera: CameraState) -> torch.Tensor:
        """Renders an image for a given camera state."""
        if self.render_state_machine is None:
            raise ValueError("RenderStateMachine is not set in RendererManager")

        # Set the render state machine to high quality
        self.render_state_machine.state = "high"

        # Render the image
        image = self.render_state_machine._render_img(camera)["rgb"].cpu().numpy()
        return image
