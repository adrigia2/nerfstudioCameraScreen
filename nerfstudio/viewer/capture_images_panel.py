import math
from typing import List
import torch
from viser import ViserServer
import viser
from PIL import Image
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.models.base_model import Model
from nerfstudio.viewer.render_panel import CameraPath, Keyframe
from nerfstudio.viewer.render_state_machine import RenderStateMachine
from nerfstudio.viewer.utils import CameraState
from nerfstudio.viewer.viewer_elements import ViewerButton, ViewerCheckbox, ViewerDropdown, ViewerNumber, ViewerSlider, ViewerText, ViewerVec3
from scipy.spatial.transform import Rotation

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
        self.g = ViewerVec3(name="3D Vector", default_value=(0.1, 0.7, 0.1), visible=True)

        with self.server.gui.add_folder("TestOptions"):
            self.a.install(self.server)
            self.b.install(self.server)
            self.c.install(self.server)
            self.d.install(self.server)
            self.e.install(self.server)
            self.f.install(self.server)
            self.g.install(self.server)

        self.base_dirs = {
            "Front":  torch.tensor([0.0, -1.0,  0.0], dtype=torch.float32),
            "Back":   torch.tensor([0.0, 1.0,  0.0], dtype=torch.float32),
            "Right":  torch.tensor([1.0,  0.0,  0.0], dtype=torch.float32),
            "Left":   torch.tensor([-1.0, 0.0,  0.0], dtype=torch.float32),
            "Top":    torch.tensor([0.0,  0.0,  1.0], dtype=torch.float32),
        }
        self.cameras = self.generate_multiple_camera_states()

    def set_render_state_machine(self, render_state_machine: RenderStateMachine) -> None:
        self.render_state_machine = render_state_machine

    def set_client(self, client: viser.ClientHandle) -> None:
        self.client = client

    def handle_btn(self) -> None:
        print("Button clicked!")
        for camera in self.cameras:
            print(f"Rendering image for camera: {camera.name}")

            self.render_state_machine.state = "high"
            image=self.render_state_machine._render_img(camera)["rgb"].cpu().numpy()
            # save image as png
            image_array = image  # È già in formato numpy grazie a .cpu().numpy()

            file_name = f"output_{camera.name}.png"
            # Converti l'array numpy in un'immagine PIL
            image_pil = Image.fromarray((image_array * 255).astype(np.uint8))  # Sa a [0, 255] se l'array è in [0, 1]
            image_pil.save(file_name)

            #read the output.png file and send it to the client
            with open(file_name, "rb") as f:
                image_bytes = f.read()
                self.client.send_file_download(file_name, image_bytes)

            print("camera_matrix:", camera.c2w)
            print("position:", camera.fov)
            print("aspect:", camera.aspect)
            

    
    def get_c2w_matrix(self, direction: str) -> torch.Tensor:
        """
        Ritorna la matrice 4x4 camera-to-world (C2W) per la telecamera
        posizionata sulla sfera unitaria (raggio=1) nella direzione
        specificata, guardando verso l'origine (0,0,0).

        :param direction: Stringa che descrive la posizione della camera,
                          ad es. "Top", "Back Right", "Front Left Top", ecc.
        :return: Matrice 4x4 di trasformazione (torch.Tensor).
        """
        # 1. Somma i vettori base corrispondenti alle parole chiave nella direzione
        #    (es. "Back Right Top" => base_dirs["Back"] + base_dirs["Right"] + base_dirs["Top"])
        words = direction.split()
        if not words:
            raise ValueError("Direzione non valida o vuota.")

        # Calcola il vettore posizione (x, y, z)
        pos = torch.zeros(3, dtype=torch.float32)
        for w in words:
            if w not in self.base_dirs:
                raise ValueError(f"La parola '{w}' non è tra le direzioni base supportate.")
            pos += self.base_dirs[w]

        # Normalizza per avere raggio 1
        norm = torch.norm(pos)
        if norm < 1e-8:
            raise ValueError(f"Il vettore posizione risulta nullo per la direzione '{direction}'.")
        pos = pos / norm  # raggio unitaria

        # 2. Definisci il target (centro sfera) e l'up world
        target = torch.zeros(3, dtype=torch.float32)  # l'origine
        world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

        # 3. Costruisci la matrice di trasformazione camera->world.
        #    Vogliamo che la camera, in *camera space*, guardi lungo l'asse +Z o -Z?
        #    Qui scegliamo che in camera space il forward locale sia +Z o -Z a seconda dello standard desiderato.
        #
        #    Esempio: se vogliamo che la camera guardi verso l'origine (dall'esterno),
        #    allora la direzione "forward" (in WORLD space) può essere (eye - target)
        #    in modo che l'asse Z in camera space punti "in avanti" dal punto di vista della camera.
        #    Se invece preferiamo che la camera guardi lungo -Z in camera space,
        #    possiamo invertire "forward" (eye - target) => - (target - eye).
        
        # In questo esempio, definiamo la Z locale della camera come (eye - target) normalizzata.
        eye = pos  # la camera è posizionata su 'pos'
        forward = eye - target
        forward = forward / torch.norm(forward)

        # Calcolo dell'asse X (right) come cross(world_up, forward) o cross(forward, world_up) a seconda delle convenzioni
        # La scelta dipende da quale verso vuoi che sia "right" in camera space (assenza di roll).
        right = torch.cross(world_up, forward)
        right_norm = torch.norm(right)
        # Se è troppo piccolo, significa che siamo quasi collineari con world_up; bisogna risolvere la degenerazione
        if right_norm < 1e-8:
            # fallback: se forward è quasi collineare con world_up, scegli un up differente
            # (per evitare una cross nulla)
            fallback_up = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            right = torch.cross(fallback_up, forward)
            right_norm = torch.norm(right)
        right = right / right_norm

        # Asse Y locale della camera ottenuto come cross(forward, right)
        # Così facendo, {right, up_cam, forward} formano una terna destrorsa
        up_cam = torch.cross(forward, right)
        up_cam = up_cam / torch.norm(up_cam)

        # Matrice di rotazione (3x3): [Rx, Ry, Rz] come colonne
        # dove Rz = forward, Ry = up_cam, Rx = right (oppure un'altra disposizione se preferisci)
        R = torch.stack([right, up_cam, forward], dim=1)  # shape (3,3)

        # Traslazione
        T = eye.unsqueeze(1)  # shape (3,1)

        # Componiamo la matrice 4x4 camera->world
        # [ R  T ]
        # [ 0  1 ]
        # Matrice 3×4: [R | T]
        c2w_3x4 = torch.cat([R, T], dim=1)  # shape (3,4)
        return c2w_3x4



    # Utility function to create camera state
    def create_camera_state(self, fov: float, aspect: float, direction: str, camera_type: CameraType) -> CameraState:
        """Creates a CameraState for a given position."""
        # Default translation for the camera to be placed 1 unit away from the origin
                # Combine rotation and translation to form c2w matrix
        

        # get camera matrix from string
        camera_matrix = self.get_c2w_matrix(direction)
        c2w = torch.tensor(camera_matrix, dtype=torch.float64)

        # Create the CameraState instance
        return CameraState(fov=fov, aspect=aspect, c2w=c2w, camera_type=
                CameraType.PERSPECTIVE
                if camera_type == "Perspective"
                else CameraType.FISHEYE
                if camera_type == "Fisheye"
                else CameraType.EQUIRECTANGULAR
                if camera_type == "Equirectangular"
                else CameraType.PERSPECTIVE,
                name=direction
                )


    def decompose_c2w(self,matrix):
        """
        Decomposes a 4x4 camera-to-world (C2W) transformation matrix into position and quaternion.
        
        Parameters:
            matrix (np.ndarray): 4x4 transformation matrix.
            
        Returns:
            position (np.ndarray): 3D position vector.
            quaternion (np.ndarray): Quaternion (x, y, z, w).
        """
        # Extract the translation (position)
        position = matrix[:3, 3]
        position *= 10.0

        # Subtract the offset (0, 0, 0.3)
        #offset = torch.tensor([0.0, 0.0, 0.5*10], dtype=position.dtype, device=position.device)
        #position -= offset

        # Extract the rotation matrix (upper-left 3x3 part)
        rotation_matrix = matrix[:3, :3]

        # Convert rotation matrix to quaternion (SciPy uses (x, y, z, w) format)
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

        return position, quaternion


    def generate_multiple_camera_states(self) -> List[CameraState]:
        # Example usage
        fov = 39.61  # Field of view in degrees
        aspect = 1  # Aspect ratio
        camera_type = CameraType.PERSPECTIVE

        directions = [
        "Top",
        "Right Top",
        "Back Right Top",
        "Back Top",
        "Back Left Top",
        "Left Top",
        "Front Left Top",
        "Front Top",
        "Front Right Top",
        "Right",
        "Back Right",
        "Back",
        "Back Left",
        "Left",
        "Front Left",
        "Front",
        "Front Right" ]

        camera_states = [self.create_camera_state(fov, aspect, direction, camera_type) for direction in directions]
        #
        #for camera_state in camera_states:
        #    positon, quaternion = self.decompose_c2w(camera_state.c2w)
        #    keyframe = Keyframe(
        #        position=positon,
        #        wxyz=quaternion,
        #        override_fov_enabled=False,
        #        override_fov_rad=camera_state.fov * np.pi / 180.0,
        #        override_time_enabled=False,
        #        override_time_val=0.0,
        #        aspect=camera_state.aspect,
        #        override_transition_enabled=False,
        #        override_transition_sec=None
        #    )
        #    self.camera_path.add_camera(keyframe)
                                    
        return camera_states


    
