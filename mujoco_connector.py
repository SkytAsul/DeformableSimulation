from engine import Engine
from openxr import OpenXrConnector
import xr
import mujoco as mj
import mujoco.viewer as mj_viewer
import math
import numpy as np

class MujocoConnector(Engine):
    def __init__(self, xml_path : str, viewer = True):
        """Creates the MuJoCo Connector with the MJCF at the passed path.

        Args:
            xml_path (str): path to the XML file containing the MJCF
            viewer (bool): whether or not to show an interactive viewer
        """
        self._viewer = viewer
        self._model = mj.MjModel.from_xml_path(xml_path)
        self._data = mj.MjData(self._model)
        self._fetch_finger()

    def _fetch_finger(self):
        finger = self._model.body("finger")
        finger_geom = self._model.geom(finger.geomadr)
        self._finger_id = finger.mocapid[0]
        self._finger_base_pos = finger.pos
        self._finger_axis_point = finger.pos.copy()
        self._finger_axis_point[1] -= finger_geom.size[1]

    def move_finger(self, angle : float):
        rot = np.array([.0, .0, .0, .0])
        mj.mju_euler2Quat(rot, [-angle, 0, 0], "xyz")
        
        mj.mju_euler2Quat(self._data.mocap_quat[self._finger_id], [math.pi/2 - angle, 0, 0], "xyz")
        mj.mju_rotVecQuat(self._data.mocap_pos[self._finger_id], self._finger_base_pos - self._finger_axis_point, rot)
        self._data.mocap_pos[self._finger_id] += self._finger_axis_point
    
    def get_contact_force(self) -> float:
        data = self._data.sensor("fingertip_sensor").data
        # data is an array containing only one number: the normal force
        return data[0] / 30

    def start_simulation(self):
        if self._viewer:
            self._viewer = mj_viewer.launch_passive(self._model, self._data)
            self._viewer.cam.azimuth = 138
            self._viewer.cam.distance = 0.5
            self._viewer.cam.elevation = -16
            self._viewer.cam.lookat = self._finger_base_pos.copy()

    def step_simulation(self):
        mj.mj_step(self._model, self._data)
        if self._viewer:
            self._viewer.sync()

    def stop_simulation(self):
        if self._viewer:
            self._viewer.close()

class MujocoRenderer:
    def __init__(self, mujoco : MujocoConnector, openxr : OpenXrConnector):
        self._mujoco = mujoco
        self._openxr = openxr

        self._scene = mj.MjvScene(mujoco._model, 1000)
    
    def init_context(self):
        self._context = mj.MjrContext(self._mujoco._model, mj.mjtFontScale.mjFONTSCALE_100)

    def _update_eye(self, eye : xr.View, camera : mj.MjvGLCamera):
        camera.pos = eye.pose.position
        # do something with orientation
        # do something with frustum

    def update_eyes(self):
        left_eye, right_eye = self._openxr.get_eyes()
        self._update_eye(left_eye, self._scene.camera[0])
        self._update_eye(right_eye, self._scene.camera[1])

    def prepare_render(self):
        mj.mjv_updateScene(self._mujoco._model, self._mujoco._data, None, None, None, mj.mjtCatBit.mjCAT_ALL, self._scene)

    def render_scene(self):
        viewport = mj.mjr_maxViewport(self._context)

        mj.mjr_render(viewport, self._scene, self._context)