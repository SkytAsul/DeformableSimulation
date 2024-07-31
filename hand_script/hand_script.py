"""
A util similar to obj2mjcf to convert a 3D hand file to a MuJoCo MJCF file.

Unlike obj2mjcf, this util recreates a hierarchy of bodies and constructs the convex
hulls on each subpart of the hand, thus making the constructed XML file almost ready-to-use.

I dropped it before the end because OBJ files do not contain the necessary data to position
joints properly and set their axis. The "skeleton_loader" script was made to complete this one but
failed, as you can read in its header.
"""

from obj2mjcf.material import Material
from obj2mjcf import constants

import trimesh
from trimesh import Geometry, Scene
import coacd

import numpy as np

from dm_control import mjcf
from dm_control.mjcf import Element

from pathlib import Path
import shutil
import logging

from dataclasses import dataclass
import tyro
from tyro.conf import Positional

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

@dataclass
class Options:
    hand_model: Positional[Path]
    """Path to the hand model (.obj)"""
    output: Positional[Path]
    """Path to the output directory"""
    overwrite: bool = False
    """Should the output directory be automatically deleted (if present).
    If not enabled, the user will be prompted.
    """
    reuse_old_assets: bool = True
    """Should the existing .obj assets from another execution be reused.
    Only effective if the user answered "Yes" to the overwrite prompt or if the overwrite option is enabled.
    """
    only_visual: bool = False
    """Skip convex decomposition and splitting in multiple bodies"""
    scale: float = 1
    """Scale factor of the model"""
    verbose: bool = True
    """Should informations be printed to the standard output."""

class HandCreator:
    def __init__(self, options: Options):
        self.options = options

        logging.basicConfig()

    def find_material_file(self) -> Path:
        with open(self.options.hand_model, "r") as f:
            for line in f.readlines():
                if line.startswith("mtllib"):
                    mat_path = self.options.hand_model.parent / line[7:-1] # no \n at the end
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

    def load_meshes(self) -> Scene:
        logging.info(f"Loading model file {self.options.hand_model}...")
        mesh = trimesh.load_mesh(
            self.options.hand_model,
            split_object=True,
            group_material=True,
            process=False,
            maintain_order=False,
        )

        if not isinstance(mesh, Scene):
            raise RuntimeError("No submeshes")
        
        return mesh

    def create_hand_part(self, fullname: str, parent: Element, convex_parts: int = -1):
        logging.info(f"Creating hand part {fullname}...")

        geom = self.meshes.geometry.get(fullname)
        if geom is None:
            raise RuntimeError("No geometry for part", fullname)
        geom: trimesh.Trimesh

        material = self.materials.get(fullname)
        if material is None:
            raise RuntimeError("No material for part", fullname)

        # Scale before recenter !
        geom.vertices *= self.options.scale

        if self.options.only_visual:
            partbody = parent
        else:
            # Recenter the geometry
            # WARNING: the center is arbitrary. Joints would have to be repositioned.
            x,y,z  = [ [ v[i] for v in geom.vertices ] for i in range(3) ]
            center = [ (max(axis) + min(axis))/2 for axis in [x,y,z] ]
            geom.vertices -= center

            # Create the body in which we will put the geometries
            partbody = parent.add("body", name=fullname, pos=relative_pos(center, parent))

        # Visual mesh, geometry and material
        visual_name = fullname + "_visual"
        self.model.asset.add("material", name=material.name,
                             specular=material.mjcf_specular(),
                             shininess=material.mjcf_shininess(),
                             rgba=material.mjcf_rgba())
        # We do not need to export the geometry to a file: by just getting the text contents,
        # we can pass them to MJCF via assets and it will take care of saving it.
        mesh_contents = geom.export(None, "obj", include_texture=True, header=None)
        self.model.asset.add("mesh", name=visual_name,
                             file=UnhashedAsset(mesh_contents, visual_name, ".obj"))
        partbody.add("geom", mesh=visual_name, name=visual_name,
                     dclass="visual", material=material.name)

        if not self.options.only_visual:
            # Collision meshes and geometries
            if self.options.reuse_old_assets:
                logging.info("Reusing convex hull from another time...")
                for collision_file in self.options.output.glob(fullname + "_collision_*"):
                    i = collision_file.stem.split("_")[-1]
                    collision_name = collision_file.stem
                    self.model.asset.add("mesh", name=collision_name,
                                        file=UnhashedAsset(collision_file.read_text(), collision_name, ".obj"))
                    colors = np.random.rand(3)
                    partbody.add("geom", mesh=collision_name, name=collision_name,
                                dclass="collision", rgba = [*colors, 1])
            else:
                logging.info("Decomposing convex hull...")
                mesh = coacd.Mesh(geom.vertices, geom.faces)
                parts = coacd.run_coacd(mesh, max_convex_hull=convex_parts)

                for i, (vs, fs) in enumerate(parts):
                    collision_name = f"{fullname}_collision_{i}"
                    part_geom = trimesh.Trimesh(vs, fs)
                    mesh_contents = part_geom.export(None, "obj", include_texture=False, header=None)
                    self.model.asset.add("mesh", name=collision_name,
                                        file=UnhashedAsset(mesh_contents, collision_name, ".obj"))
                    colors = np.random.rand(3)
                    partbody.add("geom", mesh=collision_name, name=collision_name,
                                dclass="collision", rgba = [*colors, 1])

        return partbody

    def create_finger(self, name: str, parent: Element, subparts: int):
        last_parent = parent
        for i in range(1, subparts+1):
            last_parent = self.create_hand_part(f"{name}_{i}", last_parent, convex_parts=3)

    def add_defaults(self, model: mjcf.RootElement):
        visual = model.default.add("default", dclass="visual")
        visual.geom.set_attributes(type="mesh", group=2, contype=0, conaffinity=0)

        collision = model.default.add("default", dclass="collision")
        collision.geom.set_attributes(type="mesh", group=3)

    def main(self):
        if self.options.verbose:
            logging.getLogger().setLevel(logging.INFO)

        should_remove = False
        if self.options.output.exists():
            if not self.options.overwrite and input("Output already exists. Do you want to remove ? [Y/N] ").lower() != "y":
                return
            
            if self.options.reuse_old_assets:
                should_remove = True
            else:
                shutil.rmtree(self.options.output)
        else:
            self.options.reuse_old_assets = False

        self.materials = self.load_materials(self.find_material_file())
        self.meshes = self.load_meshes()

        self.model = mjcf.RootElement(model="Hand")

        self.add_defaults(self.model)

        hand_body = self.model.worldbody.add("body", name="hand")
        self.create_finger("thumb", hand_body, 2)
        self.create_finger("index", hand_body, 3)
        self.create_finger("middle", hand_body, 3)
        self.create_finger("annular", hand_body, 3)
        self.create_finger("pinky", hand_body, 3)
        self.create_hand_part("wrist", hand_body, convex_parts=6)
        self.create_hand_part("palm", hand_body, convex_parts=7)

        if should_remove:
            shutil.rmtree(self.options.output)

        mjcf.export_with_assets(self.model, self.options.output, "Hand.xml", precision=3)

if __name__ == "__main__":
    options = tyro.cli(Options)
    HandCreator(options).main()