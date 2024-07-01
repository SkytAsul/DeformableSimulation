from engine import Engine
import mujoco
import mujoco.viewer
import math
import numpy as np

class MujocoConnector(Engine):
    def __init__(self, xml_path : str):
        """Creates the MuJoCo Connector with the MJCF at the passed path.

        Args:
            mcjf_path (str): path to the XML file containing the MJCF
        """
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
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
        mujoco.mju_euler2Quat(rot, [-angle, 0, 0], "xyz")
        
        mujoco.mju_euler2Quat(self._data.mocap_quat[self._finger_id], [math.pi/2 - angle, 0, 0], "xyz")
        mujoco.mju_rotVecQuat(self._data.mocap_pos[self._finger_id], self._finger_base_pos - self._finger_axis_point, rot)
        self._data.mocap_pos[self._finger_id] += self._finger_axis_point
    
    def get_contact_force(self) -> float:
        data = self._data.sensor("fingertip_sensor").data
        # data is an array containing only one number: the normal force
        return data[0] / 30

    def start_simulation(self):
        self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        self._viewer.cam.azimuth = 138
        self._viewer.cam.distance = 0.5
        self._viewer.cam.elevation = -16
        self._viewer.cam.lookat = self._finger_base_pos.copy()

    def step_simulation(self):
        mujoco.mj_step(self._model, self._data)
        self._viewer.sync()

    def stop_simulation(self):
        self._viewer.close()