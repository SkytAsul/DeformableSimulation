from interfaces import Engine, Visualizer
import mujoco as mj
import mujoco.viewer as mj_viewer
import math
import numpy as np

class MujocoConnector(Engine):
    def __init__(self, xml_path : str):
        """Creates the MuJoCo Connector with the MJCF at the passed path.

        Args:
            xml_path (str): path to the XML file containing the MJCF
        """
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self._fetch_finger()

    def _fetch_finger(self):
        finger = self.model.body("finger")
        finger_geom = self.model.geom(finger.geomadr)
        self._finger_id = finger.mocapid[0]
        self._finger_base_pos = finger.pos
        self._finger_axis_point = finger.pos.copy()
        self._finger_axis_point[1] -= finger_geom.size[1]

        self._right_hand_id = self.model.body("right_hand_mocap").mocapid[0]

    def move_finger(self, angle : float):
        rot = np.array([.0, .0, .0, .0])
        mj.mju_euler2Quat(rot, [-angle, 0, 0], "xyz")
        
        mj.mju_euler2Quat(self.data.mocap_quat[self._finger_id], [math.pi/2 - angle, 0, 0], "xyz")
        mj.mju_rotVecQuat(self.data.mocap_pos[self._finger_id], self._finger_base_pos - self._finger_axis_point, rot)
        self.data.mocap_pos[self._finger_id] += self._finger_axis_point
    
    def move_hand(self, hand_id: int, position: list[int], rotation: list[int]):
        self.data.mocap_pos[self._right_hand_id] = position
        self.data.mocap_quat[self._right_hand_id] = rotation

    def get_contact_force(self) -> float:
        data = self.data.sensor("fingertip_sensor").data
        # data is an array containing only one number: the normal force
        return data[0] / 30

    def step_simulation(self, duration: float | None):
        if duration is None:
            mj.mj_step(self.model, self.data)
        else:
            step_count = int(duration // self.model.opt.timestep)
            for _ in range(step_count):
                mj.mj_step(self.model, self.data)
    
    def reset_simulation(self):
        mj.mj_resetData(self.model, self.data)

class MujocoSimpleVisualizer(Visualizer):
    def __init__(self, mujoco: MujocoConnector, framerate: int | None = None):
        self._mujoco = mujoco
        self._scene = mj.MjvScene(mujoco.model, 1000)
        self._framerate = framerate
    
    def start_visualization(self):
        self._viewer = mj_viewer.launch_passive(self._mujoco.model, self._mujoco.data,
                                                show_left_ui=False, show_right_ui=False)
        self._viewer.cam.azimuth = 138
        self._viewer.cam.distance = 0.5
        self._viewer.cam.elevation = -16
        self._viewer.cam.lookat = self._mujoco._finger_base_pos.copy()

    def render_frame(self):
        self._viewer.sync()

    def should_exit(self):
        return not self._viewer.is_running()
    
    def stop_visualization(self):
        self._viewer.close()