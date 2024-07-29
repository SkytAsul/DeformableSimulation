from interfaces import Engine, Visualizer
from hand import Hand
from math import radians

import mujoco as mj
import mujoco.viewer as mj_viewer

import numpy as np
from thumb_sim import *
from index_sim import *
from middle_sim import *
from annular_sim import *
from pinky_sim import *

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
        self._fetch_finger_joints()

        self._should_reset = False
        self.model.opt.timestep = 0.001
        
        mj.mj_forward(self.model, self.data)

    def _edit_hands(self, spec: mj.MjSpec, hands: tuple[Hand, Hand]):
        if True:
            return
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

    def _fetch_finger_joints(self):
        self.joint_ids = {
            #'FreeJoint': self.model.joint('FreeJoint').id,
            'Index_J1': self.model.joint('Index_J1.stl').id,
            'Index_J2': self.model.joint('Index_J2.stl').id,
            'Index_J3': self.model.joint('Index_J3.stl').id,
            
            'Middle_J1': self.model.joint('Middle_J1.stl').id,
            'Middle_J2': self.model.joint('Middle_J2.stl').id,
            'Middle_J3': self.model.joint('Middle_J3.stl').id,
            
            #'Thumb_J1': self.model.joint('Thumb_J1.stl').id,
            'Thumb_J2': self.model.joint('Thumb_J2.stl').id,
            'Thumb_J3': self.model.joint('Thumb_J3.stl').id,
            
            'Annular_J1': self.model.joint('Annular_J1.stl').id,
            'Annular_J2': self.model.joint('Annular_J2.stl').id,
            'Annular_J3': self.model.joint('Annular_J3.stl').id,

            'Pinky_J1': self.model.joint('Pinky_J1.stl').id,
            'Pinky_J2': self.model.joint('Pinky_J2.stl').id,
            'Pinky_J3': self.model.joint('Pinky_J3.stl').id,
        }
        
    
    def init_task(self):
        # Middle
        L12 = self.data.xpos[self.model.body('Middle_J2.stl').id] - self.data.xpos[self.model.body('Middle_J1.stl').id]
        L1y_middle = np.linalg.norm([L12[1],L12[0]])
        L1z_middle =  np.linalg.norm([L12[2],L12[0]])

        L23 = self.data.xpos[self.model.body('Middle_J3.stl').id] - self.data.xpos[self.model.body('Middle_J2.stl').id]
        L2y_middle = np.linalg.norm([L23[1],L23[0]])
        L2z_middle = np.linalg.norm([L23[2],L23[0]])

        L3T = self.data.site_xpos[self.model.site('forSensorMiddle_4.stl').id] - self.data.xpos[self.model.body('Middle_J3.stl').id]
        L3y_middle = np.linalg.norm([L3T[1],L3T[0]])
        L3z_middle = np.linalg.norm([L3T[2],L3T[0]])

        self.middle_links = [L1y_middle, L1z_middle, L2y_middle, L2z_middle, L3y_middle, L3z_middle]
        
        # Annular
        L12 = self.data.xpos[self.model.body('Annular_J2.stl').id] - self.data.xpos[self.model.body('Annular_J1.stl').id]
        L1y_annular = np.linalg.norm([L12[1],L12[0]])
        L1z_annular =  np.linalg.norm([L12[2],L12[0]])

        L23 = self.data.xpos[self.model.body('Annular_J3.stl').id] - self.data.xpos[self.model.body('Annular_J2.stl').id]
        L2y_annular = np.linalg.norm([L23[1],L23[0]])
        L2z_annular = np.linalg.norm([L23[2],L23[0]])

        L3T = self.data.site_xpos[self.model.site('forSensorAnnular_4.stl').id] - self.data.xpos[self.model.body('Annular_J3.stl').id]
        L3y_annular = np.linalg.norm([L3T[1],L3T[0]])
        L3z_annular = np.linalg.norm([L3T[2],L3T[0]])
        
        self.annular_links = [L1y_annular, L1z_annular, L2y_annular, L2z_annular, L3y_annular, L3z_annular]
        
        # Pinky
        L12 = self.data.xpos[self.model.body('Pinky_J2.stl').id] - self.data.xpos[self.model.body('Pinky_J1.stl').id]
        L1y_pinky = np.linalg.norm([L12[1],L12[0]])
        L1z_pinky =  np.linalg.norm([L12[2],L12[0]])

        L23 = self.data.xpos[self.model.body('Pinky_J3.stl').id] - self.data.xpos[self.model.body('Pinky_J2.stl').id]
        L2y_pinky = np.linalg.norm([L23[1],L23[0]])
        L2z_pinky = np.linalg.norm([L23[2],L23[0]])

        L3T = self.data.site_xpos[self.model.site('forSensorPinky_4.stl').id] - self.data.xpos[self.model.body('Pinky_J3.stl').id]
        L3y_pinky= np.linalg.norm([L3T[1],L3T[0]])
        L3z_pinky = np.linalg.norm([L3T[2],L3T[0]])
        
        self.pinky_links = [L1y_pinky, L1z_pinky, L2y_pinky, L2z_pinky, L3y_pinky, L3z_pinky]
        
        # Index
        L12 = self.data.xpos[self.model.body('Index_J2.stl').id] - self.data.xpos[self.model.body('Index_J1.stl').id]
        L1y_index = np.linalg.norm([L12[1],L12[0]]) 
        L1z_index = np.linalg.norm([L12[2],L12[0]])

        L23 = self.data.xpos[self.model.body('Index_J3.stl').id] - self.data.xpos[self.model.body('Index_J2.stl').id]
        L2y_index = np.linalg.norm([L23[1],L23[0]])
        L2z_index = np.linalg.norm([L23[2],L23[0]])

        L3T = self.data.site_xpos[self.model.site('forSensor').id] - self.data.xpos[self.model.body('Index_J3.stl').id]
        L3y_index = np.linalg.norm([L3T[1],L3T[0]])
        L3z_index = np.linalg.norm([L3T[2],L3T[0]])
        
        self.index_links = [L1y_index, L1z_index, L2y_index, L2z_index, L3y_index, L3z_index]
        

        # Thumb
        L23 = self.data.xpos[self.model.body('Thumb_J3.stl').id] - self.data.xpos[self.model.body('Thumb_J2.stl').id]
        L2x_thumb = np.linalg.norm([L23[2],L23[0]])
        L2y_thumb = np.linalg.norm([L23[2],L23[1]])

        L3T = self.data.site_xpos[self.model.site('forSensorThumb_3.stl').id] - self.data.xpos[self.model.body('Thumb_J3.stl').id]
        L3x_thumb = np.linalg.norm([L3T[2],L3T[0]])
        L3y_thumb = np.linalg.norm([L3T[2],L3T[1]])

        self.thumb_links = [L2x_thumb, L2y_thumb, L3x_thumb, L3y_thumb]
        
        self.timestep = self.model.opt.timestep

        self.integral_middle = 0
        self.integral_index = 0
        self.integral_thumb = 0
        self.integral_annular = 0
        self.integral_pinky = 0
        
    

    def mapping(self, closure, finger):
        max_dist = 0
        min_dist = 0
        match finger:
            case "index": 
                max_dist = 0.11
                min_dist = 0.02
            case "middle": 
                max_dist = 0.125
                min_dist = 0.01
            case "thumb":
                max_dist = 0.1
                min_dist = 0
            case "annular": 
                max_dist = 0.12
                min_dist = 0.02
            case "pinky": 
                max_dist = 0.1
                min_dist = 0.035
        distance = max_dist - closure*(max_dist - min_dist)
        return distance

 
    def move_finger(self, hand_id: int, finger: str, closure: float):
        palm = self.data.xpos[self.model.body('right_hand_mocap').id]     
        R = self.data.xmat[self.model.body('right_hand_mocap').id]
        R = np.reshape(R, [3, 3])
        dr = self.mapping(closure, finger)
        match finger:
            case "index":
                move_index(dr,self.data,self.model,self.joint_ids,palm,self.index_links,R)
            case "middle":
                move_middle(dr,self.data,self.model,self.joint_ids,palm,self.middle_links,R)
            case "pinky":
                move_pinky(dr,self.data,self.model,self.joint_ids,palm,self.pinky_links,R)
            case "thumb":
                move_thumb(dr,self.data,self.model,self.joint_ids,palm,self.thumb_links,R)
            case "annular":
                move_annular(dr,self.data,self.model,self.joint_ids,palm,self.annular_links,R)

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