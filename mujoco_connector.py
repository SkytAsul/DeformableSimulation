from interfaces import Engine, Visualizer
from hand import Hand
from math import radians

import mujoco as mj
import mujoco.viewer as mj_viewer

class MujocoConnector(Engine):
    def __init__(self, xml_path: str, hands: tuple[Hand, Hand]):
        """Creates the MuJoCo Connector with the MJCF at the passed path.

        Args:
            xml_path (str): path to the XML file containing the MJCF
            hands (tuple[Hand, Hand]): hands configuration
        """
        spec = mj.MjSpec()
        spec.from_file(xml_path)

        self._edit_hands(spec, hands)
        
        self.model = spec.compile()
        self.data = mj.MjData(self.model)

        self._fetch_hands(hands)

        self._should_reset = False

    def _edit_hands(self, spec: mj.MjSpec, hands: tuple[Hand, Hand]):
        for hand in hands:
            hand_body = spec.find_body(f"{hand.side}_hand")

            if hand.tracking or hand.haptics:
                controller_rotation = hand.controller_rotation if spec.degree else radians(hand.controller_rotation)
                hand_body_rotation = hand_body.alt
                hand_body_rotation.euler[1] += controller_rotation
                hand_body.alt = hand_body_rotation
            else:
                # if this hand is used for neither tracking nor haptics, we can delete it from the scene.
                spec.detach_body(hand_body)
                # detach takes care of removing weld constraints, sensors and everything.

    def _fetch_hands(self, hands: tuple[Hand, Hand]):
        self._hand_mocaps = [self.model.body(f"{hand.side}_hand_mocap").mocapid[0] if hand.tracking else 0 for hand in hands]
    
    def move_hand(self, hand_id: int, position: list[float], rotation: list[float]):
        self.data.mocap_pos[self._hand_mocaps[hand_id]] = position
        self.data.mocap_quat[self._hand_mocaps[hand_id]] = rotation

    def move_finger(self, hand_id: int, finger: str, closure: float):
        # TODO insert finger movement here
        pass

    def get_contact_force(self, hand_id: int, finger: str) -> float:
        sensor_name = "left" if hand_id == 0 else "right"
        sensor_name += "_fingertip_" + finger
        # e.g. left_fingertip_thumb
        data = self.data.sensor(sensor_name).data
        # data is an array containing only one number: the normal force
        return data[0] / 30

    def step_simulation(self, duration: float | None):
        if self._should_reset:
            mj.mj_resetData(self.model, self.data)
            self._should_reset = False

        if duration is None:
            mj.mj_step(self.model, self.data)
        else:
            step_count = int(duration // self.model.opt.timestep)
            for _ in range(step_count):
                mj.mj_step(self.model, self.data)
    
    def reset_simulation(self):
        self._should_reset = True

class MujocoSimpleVisualizer(Visualizer):
    def __init__(self, mujoco: MujocoConnector, framerate: int | None = None):
        self._mujoco = mujoco
        self._scene = mj.MjvScene(mujoco.model, 1000)
        self._framerate = framerate
    
    def start_visualization(self):
        self._viewer = mj_viewer.launch_passive(self._mujoco.model, self._mujoco.data,
                                                show_left_ui=False, show_right_ui=False)
        self._viewer.cam.azimuth = 138
        self._viewer.cam.distance = 3
        self._viewer.cam.elevation = -16

    def render_frame(self):
        self._viewer.sync()

    def should_exit(self):
        return not self._viewer.is_running()
    
    def stop_visualization(self):
        self._viewer.close()