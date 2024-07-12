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

    def create_finger_part(self, name: str, part_id: int, parent: Element):
        fullname = f"{name}_{part_id}"
        logging.info(f"Creating finger part {fullname}...")

        geom = self.meshes.geometry.get(fullname)
        if geom is None:
            raise RuntimeError("No geometry for finger", fullname)
        
        material = self.materials.get(fullname)
        if material is None:
            raise RuntimeError("No material for finger", fullname)

        # Recenter the geometry
        x,y,z  = [ [ v[i] for v in geom.vertices ] for i in range(3) ]
        center = [ (max(axis) + min(axis))/2 for axis in [x,y,z] ]
        geom.vertices -= center

        mesh_file = self.output / (fullname + ".obj")
        geom.export(mesh_file.as_posix(), include_texture=True, header=None)


        # TODO decomposition
        self.model.asset.add("mesh", file=mesh_file.name)
        self.model.asset.add("material", name=material.name,
                    specular=material.mjcf_specular(),
                    shininess=material.mjcf_shininess(),
                    rgba=material.mjcf_rgba())

        fingerbody = parent.add("body", name=fullname, pos=center)
        fingerbody.add("geom", mesh=fullname, material=fullname)

        return fingerbody

    def create_finger(self, name: str, parent: Element, subparts: int):
        last_parent = parent
        for i in range(1, subparts+1):
            last_parent = self.create_finger_part(name, i, last_parent)

    def main(self, hand_model_path: Path):
        if self.output.exists():
            res = input("Output already exists. Do you want to remove ? [Y/N]")
            if res.lower() != "y":
                return
            shutil.rmtree(self.output)

        self.output.mkdir(parents=True)

        self.materials = self.load_materials(self.find_material_file(hand_model_path))
        self.meshes = self.load_meshes(hand_model_path)

        self.model = mjcf.RootElement(model="Hand")

        hand_body = self.model.worldbody.add("body", name="hand")
        self.create_finger("thumb", hand_body, 2)
        self.create_finger("index", hand_body, 3)
        self.create_finger("middle", hand_body, 3)
        self.create_finger("annular", hand_body, 3)
        self.create_finger("pinky", hand_body, 3)

        mjcf.export_with_assets(self.model, self.output, "Hand.xml")

def generate_hand(hand_model: Path, output: Path):
    HandCreator(output).main(hand_model)

if __name__ == "__main__":
    # HandCreator(Path("./output/")).main(Path("Hand_Decomposed.obj"))
    tyro.cli(tyro.conf.Positional[generate_hand])