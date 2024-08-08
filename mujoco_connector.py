from fingers_sim import annular_sim, index_sim, middle_sim, pinky_sim, thumb_sim
from interfaces import Engine, Visualizer
from hand import Hand
from math import radians

import mujoco as mj
import mujoco.viewer as mj_viewer

from weart import TextureType

import numpy as np
from fingers_sim import *

TEXTURE_MAPPING = {
    "left kidney": TextureType.ProfiledRubberSlow,
    "liver cibrosis": TextureType.CrushedRock,
    "liver lesion": TextureType.VenetianGranite
}

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
        self._fetch_flexes(TEXTURE_MAPPING)

        self._fetch_finger_joints(hands)
        self._init_task()

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
    
    def _fetch_flexes(self, flex_textures: dict[str, TextureType]):
        self._flex_textures = {}
        for flex, texture in flex_textures.items():
            flex_id = self._get_flex_id(flex)
            if flex_id is not None:
                self._flex_textures[flex_id] = texture

    def _get_flex_id(self, flex_name: str):
        for id, name_adr in enumerate(self.model.name_flexadr):
            name_binary: bytes = self.model.names[name_adr:]
            name_decoded = name_binary.decode()
            name_decoded = name_decoded[:name_decoded.index("\0")]
            if name_decoded == flex_name:
                return id
        return None

    def _fetch_finger_joints(self, hands: tuple[Hand, Hand]):
        self.joint_ids ={}
        for hand in filter(lambda h: h.tracking or h.haptics, hands):
            for finger in ["Index_", "Middle_", "Annular_", "Pinky_", "Thumb_"]:
                for joint in ["J1","J2","J3"]:
                    name = hand.side.capitalize() + "_" + finger + joint
                    self.joint_ids[name] = self.model.actuator(name + ".stl").id
    
    def _init_task(self):
        self.middle_links = [0.030836556725044706, 0.007652464090533103, 0.021260243533355868, 0.00577589647566514, 0.008874814927647797, 0.002068895357431296]
        self.annular_links = [0.028450942497569385, 0.00801985099612217, 0.018776141709360844, 0.005880719130344506, 0.008045547312644435, 0.002237504762006266] 
        self.pinky_links = [0.019719324227772114, 0.00533552696553951, 0.013602577880681295, 0.003827636868878808, 0.005663897513197083, 0.0015979126384130552]
        self.index_links = [0.02592721506448389, 0.006917977811470749, 0.017868637273446453, 0.005247156468983898, 0.0074581157647223475, 0.001764839822760083]
        self.thumb_links = [0.0057328924064646465, 0.02297047222727691, 0.0016093799740583314, 0.007174085230947566, 0.064886404284719, 0.016350265551556033]
        self.timestep = self.model.opt.timestep
        self.maxForce = 0

    def move_hand(self, hand_id: int, position: list[float], rotation: list[float]):
        self.data.mocap_pos[self._hand_mocaps[hand_id]] = position
        self.data.mocap_quat[self._hand_mocaps[hand_id]] = rotation

    def _finger_value_mapping(self, closure, finger):
        max_dist = 0
        min_dist = 0
        match finger:
            case "index": 
                max_dist = 0.11
                min_dist = 0.04
            case "middle": 
                max_dist = 0.125
                min_dist = 0.03
            case "thumb":
                max_dist = 0.1
                min_dist = 0.064
            case "thumb_abd":
                max_dist = 0.0
                min_dist = -0.07
            case "annular": 
                max_dist = 0.12
                min_dist = 0.04
            case "pinky": 
                max_dist = 0.1
                min_dist = 0.045
        distance = max_dist - closure*(max_dist - min_dist)
        return distance

    def move_finger(self, hand_id: int, finger: str, closure: float, abduction: float):
        if finger == "middle":
            self.move_finger(hand_id, "annular", closure, abduction)
            self.move_finger(hand_id, "pinky", closure, abduction)

        if hand_id == 1:
            hand_side = 1 
            hand = "Right"
        else: 
            hand_side = -1 
            hand = "Left"
        palm = self.data.xpos[self.model.body(hand+"_Palm_"+hand).id]     
        R = self.data.xmat[self.model.body(hand+"_Palm_"+hand).id]
        R = np.reshape(R, [3, 3])
        dr = self._finger_value_mapping(closure, finger)

        hand = hand + "_"
        match finger:
            case "index":
                index_sim.move_index(dr,self.data,self.model,self.joint_ids,palm,self.index_links,R,hand)
            case "middle":
                middle_sim.move_middle(dr,self.data,self.model,self.joint_ids,palm,self.middle_links,R,hand)
            case "pinky":
                pinky_sim.move_pinky(dr,self.data,self.model,self.joint_ids,palm,self.pinky_links,R,hand)
            case "thumb":
                abd = self._finger_value_mapping(abduction,"thumb_abd")
                thumb_sim.move_thumb(dr,abd,self.data,self.model,self.joint_ids,palm,self.thumb_links,R, hand_side)
            case "annular":
                annular_sim.move_annular(dr,self.data,self.model,self.joint_ids,palm,self.annular_links,R,hand)

    def get_contact(self, hand_id: int, finger: str) -> tuple[float, TextureType | None]:
        sensor_name = "left" if hand_id == 0 else "right"
        sensor_name += "_fingertip_" + finger
        # e.g. left_fingertip_thumb


        data = self.data.sensor(sensor_name).data
        # data is an array containing only one number: the normal force
        force = data[0] / 30


        texture = None
        site_id = self.model.sensor(sensor_name).objid[0]
        sensor_body = self.model.body(self.model.site(site_id).bodyid[0])
        body_first_geom = sensor_body.geomadr[0]
        body_last_geom = body_first_geom + sensor_body.geomnum[0]

        for contact in self.data.contact:
            flex = -1
            geom = -1
            if contact.flex[0] != -1:
                flex = contact.flex[0]
                geom = contact.geom[1]
            elif contact.flex[1] != -1:
                flex = contact.flex[1]
                geom = contact.geom[0]

            if geom != -1:
                # means we have a contact between a flex and a geom
                if geom in range(body_first_geom, body_last_geom + 1):
                    # contact with the body of the sensor
                    if flex in self._flex_textures:
                        texture = self._flex_textures[flex]
    

        return (force, texture)

    def step_simulation(self, duration: float | None):
        if self._should_reset:
            mj.mj_resetData(self.model, self.data)
            self._should_reset = False

        if duration is None:
            mj.mj_step(self.model, self.data)
        else:
            step_count = int(duration // self.model.opt.timestep)
            mj.mj_step(self.model, self.data, nstep=step_count)
    
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