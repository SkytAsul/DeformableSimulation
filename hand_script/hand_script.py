from obj2mjcf.material import Material
from obj2mjcf import constants

import trimesh
from trimesh import Geometry, Scene

from dm_control import mjcf
from dm_control.mjcf import Element

from pathlib import Path
import shutil
import logging

import tyro

# All our assets have unique names so we do not need the hash in the filename
class UnhashedAsset(mjcf.Asset):
    def __init__(self, contents, name, extension):
        super().__init__(contents, extension)
        self.name = name

    def get_vfs_filename(self):
        return self.name + self.extension

def relative_pos(position: list[int], parent: Element):
    if "pos" not in parent.get_attributes():
        return position
    
    new_position = [ position[i] - parent.pos[i] for i in range(3) ]
    if parent.parent is not None:
        return relative_pos(new_position, parent.parent)
    else:
        return new_position
    
class HandCreator:
    def __init__(self, output: Path):
        self.output = output

        logging.basicConfig()

    def find_material_file(self, hand_model: Path) -> Path:
        with open(hand_model, "r") as f:
            for line in f.readlines():
                if line.startswith("mtllib"):
                    mat_path = hand_model.parent / line[7:-1] # no \n at the end
                    if mat_path.is_file():
                        logging.info(f"Found material file {mat_path}")
                        return mat_path
                    raise RuntimeError("Cannot find material file", mat_path)
        raise RuntimeError("No materials")

    def load_materials(self, materials_file: Path):
        # Stolen from obj2mjcf
        sub_mtls: list[list[str]] = []
        mtls: dict[Material] = {}

        # Parse the MTL file into separate materials.
        with open(materials_file, "r") as f:
            lines = f.readlines()
        # Remove comments.
        lines = [
            line for line in lines if not line.startswith(constants.MTL_COMMENT_CHAR)
        ]
        # Remove empty lines.
        lines = [line for line in lines if line.strip()]
        # Remove trailing whitespace.
        lines = [line.strip() for line in lines]
        # Split at each new material definition.
        for line in lines:
            if line.startswith("newmtl"):
                sub_mtls.append([])
            sub_mtls[-1].append(line)
        for sub_mtl in sub_mtls:
            material = Material.from_string(sub_mtl)
            mtls[material.name] = material

        # TODO: WE DO NOT PROCESS MATERIALS FOR NOW
        # If needed in the future (e.g. for textures), see obj2mjcf
        logging.info("Done processing MTL file.")

        return mtls

    def load_meshes(self, hand_model: Path) -> Scene:
        logging.info(f"Loading model file {hand_model}...")
        mesh = trimesh.load_mesh(
            hand_model,
            split_object=True,
            group_material=True,
            process=False,
            maintain_order=False,
        )

        if not isinstance(mesh, Scene):
            raise RuntimeError("No submeshes")
        
        return mesh

    def create_hand_part(self, fullname: str, parent: Element):
        logging.info(f"Creating hand part {fullname}...")

        geom = self.meshes.geometry.get(fullname)
        if geom is None:
            raise RuntimeError("No geometry for part", fullname)
        geom: trimesh.Trimesh

        material = self.materials.get(fullname)
        if material is None:
            raise RuntimeError("No material for part", fullname)

        # Recenter the geometry
        x,y,z  = [ [ v[i] for v in geom.vertices ] for i in range(3) ]
        center = [ (max(axis) + min(axis))/2 for axis in [x,y,z] ]
        geom.vertices -= center

        # We do not need to export the geometry to a file: by just getting the text contents,
        # we can pass them to MJCF via assets and it will take care of saving it.
        mesh_contents = geom.export(None, "obj", include_texture=True, header=None)

        # TODO decomposition
        visual_name = fullname + "-visual"
        collision_name = fullname + "-collision"
        self.model.asset.add("mesh", file=UnhashedAsset(mesh_contents, visual_name, ".obj"), name=visual_name)
        self.model.asset.add("material", name=material.name,
                    specular=material.mjcf_specular(),
                    shininess=material.mjcf_shininess(),
                    rgba=material.mjcf_rgba())

        partbody = parent.add("body", name=fullname, pos=relative_pos(center, parent))
        partbody.add("geom", mesh=visual_name, name=visual_name, dclass="visual", material=material.name)
        partbody.add("geom", mesh=visual_name, name=collision_name, dclass="collision")

        return partbody

    def create_finger(self, name: str, parent: Element, subparts: int):
        last_parent = parent
        for i in range(1, subparts+1):
            last_parent = self.create_hand_part(f"{name}_{i}", last_parent)

    def add_defaults(self, model: mjcf.RootElement):
        visual = model.default.add("default", dclass="visual")
        visual.geom.set_attributes(type="mesh", group=2, contype=0, conaffinity=0)

        collision = model.default.add("default", dclass="collision")
        collision.geom.set_attributes(type="mesh", group=3)

    def main(self, hand_model_path: Path, overwrite = False):
        if self.output.exists():
            if not overwrite and input("Output already exists. Do you want to remove ? [Y/N] ").lower() != "y":
                return
            shutil.rmtree(self.output)

        # self.output.mkdir(parents=True)
        # MJCF export takes care of that

        self.materials = self.load_materials(self.find_material_file(hand_model_path))
        self.meshes = self.load_meshes(hand_model_path)

        self.model = mjcf.RootElement(model="Hand")

        self.add_defaults(self.model)

        hand_body = self.model.worldbody.add("body", name="hand")
        self.create_finger("thumb", hand_body, 2)
        self.create_finger("index", hand_body, 3)
        self.create_finger("middle", hand_body, 3)
        self.create_finger("annular", hand_body, 3)
        self.create_finger("pinky", hand_body, 3)
        self.create_hand_part("wrist", hand_body)
        self.create_hand_part("palm", hand_body)

        mjcf.export_with_assets(self.model, self.output, "Hand.xml", precision=3)

def generate_hand(hand_model: Path, output: Path, /, overwrite: bool = False):
    HandCreator(output).main(hand_model, overwrite)

if __name__ == "__main__":
    tyro.cli(generate_hand)