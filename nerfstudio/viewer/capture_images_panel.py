from viser import ViserServer
from nerfstudio.models.base_model import Model
from nerfstudio.viewer.viewer_elements import ViewerButton, ViewerCheckbox, ViewerDropdown, ViewerNumber, ViewerSlider, ViewerText, ViewerVec3


class CaptureImagesPanel:
    def __init__(
            self,
            server : ViserServer
    ):
        self.server = server

        self.b = ViewerNumber(name="Number", default_value=1.0)
        self.c = ViewerCheckbox(name="Checkbox", default_value=False)
        self.d = ViewerDropdown(name="Dropdown", default_value="A", options=["A", "B"])
        self.e = ViewerSlider(name="Slider", default_value=0.5, min_value=0.0, max_value=1.0)
        self.f = ViewerText(name="Text", default_value="Hello World")
        self.g = ViewerVec3(name="3D Vector", default_value=(0.1, 0.7, 0.1))

        with self.server.gui.add_folder("TestOptions"):
            self.b.install(self.server)
            self.c.install(self.server)
            self.d.install(self.server)
            self.e.install(self.server)
            self.f.install(self.server)
            self.g.install(self.server)

        