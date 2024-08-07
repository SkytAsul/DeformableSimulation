"""
A class made to load a .dae file (Collada) and extract the required informations
to recreate the skeleton in the MuJoCo MJCF format. It was initially made to convert
a hand model to MuJoCo with all the joints at the right place automatically.

This script is a completion of the "hand_script".

I dropped it before the end because Collada files do not contain leaf joints, thus
our converted hand would have lacked a joint for the last part of each finger. This is
because to add a joint, we need to know the direction from one joint to the other.

It was also a mess to get the orientations right. Collada files are terrible.
"""

from collada import Collada
from collada.scene import Node, ControllerNode, SceneNode, Scene
from collada.controller import Skin

from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt

@dataclass
class Joint:
    sid: str
    matrix: list[list[float]]
    inverse_matrix: list[list[float]] | None
    children: list['Joint']

    matrix_world: list[list[float]] = None

    worldpos: list[float] = None

    @property
    def basepos(self) -> list[float]:
        return self.matrix_world.T[3, :3]
    
    @property
    def is_valid(self) -> bool:
        return self.worldpos is not None

@dataclass
class Bone:
    origin: Joint
    tail: Joint

    @property
    def name(self) -> str:
        return f"{self.origin.sid} to {self.tail.sid}"

    @property
    def length(self) -> float:
        return np.linalg.norm(self.tail.basepos - self.origin.basepos)

def find_skeleton_root(node: SceneNode) -> str:
    if isinstance(node, ControllerNode) and isinstance(node.controller, Skin):
        for children in node.xmlnode.getchildren():
            if children.tag == "{http://www.collada.org/2005/11/COLLADASchema}skeleton":
                return children.text[1:]
        raise RuntimeError("No skeleton")
    elif isinstance(node, Node):
        for child in node.children:
            root = find_skeleton_root(child)
            if root is not None:
                return root
        
def find_node(node: SceneNode, id: str, prev_mat: list[list[float]]) -> tuple[Node | None, list[list[float]] | None]:
    # shitty matrix do not use
    if node.id == id:
        return node, prev_mat
    elif isinstance(node, Node):
        for child in node.children:
            n, mat = find_node(child, id, prev_mat @ node.matrix)
            if n is not None:
                return (n, mat)

def explore_joints_tree(node: SceneNode, skin: Skin) -> Joint:
    if isinstance(node, Node):
        sid = node.xmlnode.get("sid")
        return Joint(
            sid = sid,
            matrix = node.matrix,
            inverse_matrix = skin.joint_matrices.get(sid), # can be None
            children = [j for j in map(lambda x: explore_joints_tree(x, skin), node.children) if j is not None],
        )

def concat_matrix(node: SceneNode, parent: list[list[float]], until: SceneNode):
    # ultra shitty matrix 
    if node == until:
        return True
    if isinstance(node, Node):
        mat = parent @ node.matrix
        for child in node.children:
            r = concat_matrix(child, mat, until)
            if r is True:
                return mat
            elif r is False:
                pass
            else:
                return r
    return False

def load_skeleton_joints(skin: Skin, scene: Scene) -> tuple[Joint, list[list[float]]]:
    skeleton_root = None
    for node in scene.nodes:
        skeleton_root = find_skeleton_root(node)
        if skeleton_root is not None:
            break
    
    if skeleton_root is None:
        raise RuntimeError("Cannot find skeleton root name")

    skeleton_root_node = None
    skeleton_root_transform = None
    for node in scene.nodes:
        skeleton_root_node, skeleton_root_transform = find_node(node, skeleton_root, np.identity(4))
        if skeleton_root_node is not None:
            break

    if skeleton_root_node is None:
        raise RuntimeError("Cannot find skeleton root node", skeleton_root)

    mat = None
    for node in scene.nodes:
        mat = concat_matrix(node, np.identity(4), skeleton_root_node)
        if isinstance(mat, bool):
            pass
        else:
            break

    return explore_joints_tree(skeleton_root_node, skin), mat

def _add_bones(joint: Joint, bones: list[Bone]):
    for subjoint in joint.children:
        bones.append(Bone(joint, subjoint))
        _add_bones(subjoint, bones)

def infer_bones(root_joint: Joint) -> list[Bone]:
    bones = []
    _add_bones(root_joint, bones)
    return bones

def add_joints_list(joint: Joint, list: list[Joint]):
    list.append(joint)
    for subjoint in joint.children:
        add_joints_list(subjoint, list)

def derive_matrixes(joint: Joint, prev: Joint):
    if prev is not None:
        joint.matrix_world = prev.matrix_world @ joint.matrix
    for subjoint in joint.children:
        derive_matrixes(subjoint, joint)

def derive_vertexes(joint: Joint, bind_shape_matrix: list[list[float]]):
    vert = joint.matrix_world.T[3]
    # vert = np.array([0, 0, 0, 1])
    if joint.inverse_matrix is not None:
        joint.worldpos = np.dot(joint.matrix_world @ joint.inverse_matrix @ bind_shape_matrix, vert)[:3]
    for subjoint in joint.children:
        derive_vertexes(subjoint, bind_shape_matrix)

model_path = "/data/Documents/Ecole/INSA/Info/Stage/3D/hand/testB.dae"
c = Collada(model_path)
skin = None
for controller in c.controllers:
    if isinstance(controller, Skin):
        skin = controller

if skin is None:
    raise RuntimeError("Skin controller not found")

root_joint, root_mat = load_skeleton_joints(skin, c.scene)
# print(root_joint, root_mat)

#root_joint.matrix_world = np.identity(4)
root_joint.matrix_world = root_mat @ root_joint.matrix
derive_matrixes(root_joint, None)
derive_vertexes(root_joint, skin.bind_shape_matrix)

joints_list = []
add_joints_list(root_joint, joints_list)

bones = infer_bones(root_joint)
for b in bones:
    print(f"{b.name} : {b.length}")

def viz_skeleton(bones):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    xs, ys, zs = [[joint.worldpos[i] for joint in joints_list if joint.is_valid] for i in range(3)]
    ax.scatter(xs, ys, zs)

    for joint in joints_list:
        if joint.is_valid:
            ax.text(*joint.worldpos, joint.sid)

    for bone in bones:
        if bone.origin.is_valid and bone.tail.is_valid:
            ax.plot(*[[bone.origin.worldpos[i], bone.tail.worldpos[i]] for i in range(3)])

    plt.show()

viz_skeleton(bones)


from dm_control import mjcf
from dm_control.mjcf import Element
import dm_control.mjcf.skin
import mujoco
import trimesh
model = mjcf.RootElement(model="HandBones")

def add_joint(body: Element, joint: Joint):
    quat = np.zeros(4)
    mat = joint.matrix_world
    mujoco.mju_mat2Quat(quat, mat[:3, :3].reshape(9))
    sb = body.add("body", name=joint.sid, pos=joint.worldpos, quat=quat)
    sb.add("geom", type="sphere", size="0.01")

for joint in joints_list:
    if joint.is_valid:
        add_joint(model.worldbody, joint)

# mesh = trimesh.load_mesh(model_path)
# mesh: trimesh.Scene
# geom = mesh.geometry[0]
# geom: trimesh.Trimesh

# sbones = []
# for bone in bones:
#     pass

# dm_control.mjcf.skin.serialize(vertices = geom.vertices, texcoords = [], faces = geom.faces, bones = sbones)

mjcf.export_with_assets(model, ".", "Hand.xml", precision=3)