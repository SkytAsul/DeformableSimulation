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

        opt = mj.MjvOption()
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
        self.data.mocap_quat[self._hand_mocaps[hand_id]] = rotation #quaternion_multiply(rotation, [ 0.5003982, 0.4996018, -0.4999998, -0.4999998 ])

    def _fetch_finger_joints(self):
        self.joint_ids ={}
        for hand in ["Right_", "Left_"]:
            for finger in ["Index_", "Middle_", "Annular_", "Pinky_", "Thumb_"]:
                for joint in ["J1","J2","J3"]:
                    name = hand + finger + joint
                    self.joint_ids[name] = self.model.actuator(name + ".stl").id
        # self.joint_ids = {
        #     #'FreeJoint': self.model.joint('FreeJoint').id,
        #     'Right_Index_J1': self.model.actuator('Right_Index_J1.stl').id,
        #     'Right_Index_J2': self.model.actuator('Right_Index_J2.stl').id,
        #     'Right_Index_J3': self.model.actuator('Right_Index_J3.stl').id,
            
        #     'Right_Middle_J1': self.model.actuator('Right_Middle_J1.stl').id,
        #     'Right_Middle_J2': self.model.actuator('Right_Middle_J2.stl').id,
        #     'Right_Middle_J3': self.model.actuator('Right_Middle_J3.stl').id,

        #     'Right_Annular_J1': self.model.actuator('Right_Annular_J1.stl').id,
        #     'Right_Annular_J2': self.model.actuator('Right_Annular_J2.stl').id,
        #     'Right_Annular_J3': self.model.actuator('Right_Annular_J3.stl').id,

        #     'Right_Pinky_J1': self.model.actuator('Right_Pinky_J1.stl').id,
        #     'Right_Pinky_J2': self.model.actuator('Right_Pinky_J2.stl').id,
        #     'Right_Pinky_J3': self.model.actuator('Right_Pinky_J3.stl').id,
            
        #     'Right_Thumb_J1': self.model.actuator('Right_Thumb_J1.stl').id,
        #     'Right_Thumb_J2': self.model.actuator('Right_Thumb_J2.stl').id,
        #     'Right_Thumb_J3': self.model.actuator('Right_Thumb_J3.stl').id,
            
        #     'Left_Index_J1': self.model.actuator('Left_Index_J1.stl').id,
        #     'Left_Index_J2': self.model.actuator('Left_Index_J2.stl').id,
        #     'Left_Index_J3': self.model.actuator('Left_Index_J3.stl').id,
            
        #     'Left_Middle_J1': self.model.actuator('Left_Middle_J1.stl').id,
        #     'Left_Middle_J2': self.model.actuator('Left_Middle_J2.stl').id,
        #     'Left_Middle_J3': self.model.actuator('Left_Middle_J3.stl').id,

        #     'Left_Annular_J1': self.model.actuator('Left_Annular_J1.stl').id,
        #     'Left_Annular_J2': self.model.actuator('Left_Annular_J2.stl').id,
        #     'Left_Annular_J3': self.model.actuator('Left_Annular_J3.stl').id,

        #     'Left_Pinky_J1': self.model.actuator('Left_Pinky_J1.stl').id,
        #     'Left_Pinky_J2': self.model.actuator('Left_Pinky_J2.stl').id,
        #     'Left_Pinky_J3': self.model.actuator('Left_Pinky_J3.stl').id,
            
        #     'Left_Thumb_J1': self.model.actuator('Left_Thumb_J1.stl').id,
        #     'Left_Thumb_J2': self.model.actuator('Left_Thumb_J2.stl').id,
        #     'Left_Thumb_J3': self.model.actuator('Left_Thumb_J3.stl').id
        # }

        
        
    
    def init_task(self):
        # Middle
        # L12 = self.data.xpos[self.model.body('Middle_J2.stl').id] - self.data.xpos[self.model.body('Middle_J1.stl').id]
        # L1y_middle = np.linalg.norm([L12[1],L12[0]])
        # L1z_middle =  np.linalg.norm([L12[2],L12[0]])

        # L23 = self.data.xpos[self.model.body('Middle_J3.stl').id] - self.data.xpos[self.model.body('Middle_J2.stl').id]
        # L2y_middle = np.linalg.norm([L23[1],L23[0]])
        # L2z_middle = np.linalg.norm([L23[2],L23[0]])

        # L3T = self.data.site_xpos[self.model.site('forSensorMiddle_4.stl').id] - self.data.xpos[self.model.body('Middle_J3.stl').id]
        # L3y_middle = np.linalg.norm([L3T[1],L3T[0]])
        # L3z_middle = np.linalg.norm([L3T[2],L3T[0]])

        # self.middle_links = [L1y_middle, L1z_middle, L2y_middle, L2z_middle, L3y_middle, L3z_middle]
        self.middle_links = [0.030836556725044706, 0.007652464090533103, 0.021260243533355868, 0.00577589647566514, 0.008874814927647797, 0.002068895357431296]
        # Annular
        # L12 = self.data.xpos[self.model.body('Annular_J2.stl').id] - self.data.xpos[self.model.body('Annular_J1.stl').id]
        # L1y_annular = np.linalg.norm([L12[1],L12[0]])
        # L1z_annular =  np.linalg.norm([L12[2],L12[0]])

        # L23 = self.data.xpos[self.model.body('Annular_J3.stl').id] - self.data.xpos[self.model.body('Annular_J2.stl').id]
        # L2y_annular = np.linalg.norm([L23[1],L23[0]])
        # L2z_annular = np.linalg.norm([L23[2],L23[0]])

        # L3T = self.data.site_xpos[self.model.site('forSensorAnnular_4.stl').id] - self.data.xpos[self.model.body('Annular_J3.stl').id]
        # L3y_annular = np.linalg.norm([L3T[1],L3T[0]])
        # L3z_annular = np.linalg.norm([L3T[2],L3T[0]])
        
        # self.annular_links = [L1y_annular, L1z_annular, L2y_annular, L2z_annular, L3y_annular, L3z_annular]
        self.annular_links = [0.028450942497569385, 0.00801985099612217, 0.018776141709360844, 0.005880719130344506, 0.008045547312644435, 0.002237504762006266] 
        # Pinky
        # L12 = self.data.xpos[self.model.body('Pinky_J2.stl').id] - self.data.xpos[self.model.body('Pinky_J1.stl').id]
        # L1y_pinky = np.linalg.norm([L12[1],L12[0]])
        # L1z_pinky =  np.linalg.norm([L12[2],L12[0]])

        # L23 = self.data.xpos[self.model.body('Pinky_J3.stl').id] - self.data.xpos[self.model.body('Pinky_J2.stl').id]
        # L2y_pinky = np.linalg.norm([L23[1],L23[0]])
        # L2z_pinky = np.linalg.norm([L23[2],L23[0]])

        # L3T = self.data.site_xpos[self.model.site('forSensorPinky_4.stl').id] - self.data.xpos[self.model.body('Pinky_J3.stl').id]
        # L3y_pinky= np.linalg.norm([L3T[1],L3T[0]])
        # L3z_pinky = np.linalg.norm([L3T[2],L3T[0]])
        
        # self.pinky_links = [L1y_pinky, L1z_pinky, L2y_pinky, L2z_pinky, L3y_pinky, L3z_pinky]
        self.pinky_links = [0.019719324227772114, 0.00533552696553951, 0.013602577880681295, 0.003827636868878808, 0.005663897513197083, 0.0015979126384130552]
        # Index
        # L12 = self.data.xpos[self.model.body('Index_J2.stl').id] - self.data.xpos[self.model.body('Index_J1.stl').id]
        # L1y_index = np.linalg.norm([L12[1],L12[0]]) 
        # L1z_index = np.linalg.norm([L12[2],L12[0]])

        # L23 = self.data.xpos[self.model.body('Index_J3.stl').id] - self.data.xpos[self.model.body('Index_J2.stl').id]
        # L2y_index = np.linalg.norm([L23[1],L23[0]])
        # L2z_index = np.linalg.norm([L23[2],L23[0]])

        # L3T = self.data.site_xpos[self.model.site('forSensor').id] - self.data.xpos[self.model.body('Index_J3.stl').id]
        # L3y_index = np.linalg.norm([L3T[1],L3T[0]])
        # L3z_index = np.linalg.norm([L3T[2],L3T[0]])
        
        # self.index_links = [L1y_index, L1z_index, L2y_index, L2z_index, L3y_index, L3z_index]
        self.index_links = [0.02592721506448389, 0.006917977811470749, 0.017868637273446453, 0.005247156468983898, 0.0074581157647223475, 0.001764839822760083]

        # Thumb
        # L12 = self.data.xpos[self.model.body('Thumb_J2.stl').id] - self.data.xpos[self.model.body('Thumb_J1.stl').id]
        # L1x_thumb = np.linalg.norm([L12[2],L12[0]])
        # L1y_thumb = np.linalg.norm([L12[2],L12[1]])


        # L23 = self.data.xpos[self.model.body('Thumb_J3.stl').id] - self.data.xpos[self.model.body('Thumb_J2.stl').id]
        # L2x_thumb = np.linalg.norm([L23[2],L23[0]])
        # L2y_thumb = np.linalg.norm([L23[2],L23[1]])

        # L3T = self.data.site_xpos[self.model.site('forSensorThumb_3.stl').id] - self.data.xpos[self.model.body('Thumb_J3.stl').id]
        # L3x_thumb = np.linalg.norm([L3T[2],L3T[0]])
        # L3y_thumb = np.linalg.norm([L3T[2],L3T[1]])

        # L_abd = self.data.site_xpos[self.model.site('forSensorThumb_3.stl').id] - self.data.xpos[self.model.body('Thumb_J1.stl').id]
        # Ly_abd = np.linalg.norm([L_abd[0],L_abd[1]])
        # Lz_abd = np.linalg.norm([L_abd[0],L_abd[2]])

        # print(L1x_thumb, L1y_thumb, L2x_thumb, L2y_thumb, L3x_thumb, L3y_thumb, Ly_abd, Lz_abd )
        # self.thumb_links = [L2x_thumb, L2y_thumb, L3x_thumb, L3y_thumb, Ly_abd, Lz_abd]
        self.thumb_links = [0.0057328924064646465, 0.02297047222727691, 0.0016093799740583314, 0.007174085230947566, 0.064886404284719, 0.016350265551556033]
        self.timestep = self.model.opt.timestep

        self.maxForce = 0

    

    def mapping(self, closure, finger):
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
        
        if hand_id == 1:
            hand_side = 1 
            hand = "Right"
        else: 
            hand_side = -1 
            hand = "Left"

        palm = self.data.xpos[self.model.body(hand+"_Palm_"+hand).id]     
        R = self.data.xmat[self.model.body(hand+"_Palm_"+hand).id]
        R = np.reshape(R, [3, 3])
        dr = self.mapping(closure, finger)

        hand = hand + "_"
        if finger == "thumb":
            abd = self.mapping(abduction,"thumb_abd")
        match finger:
            case "index":
                move_index(dr,self.data,self.model,self.joint_ids,palm,self.index_links,R,hand)
            case "middle":
                move_middle(dr,self.data,self.model,self.joint_ids,palm,self.middle_links,R,hand)
            case "pinky":
                move_pinky(dr,self.data,self.model,self.joint_ids,palm,self.pinky_links,R,hand)
            case "thumb":
                move_thumb(dr,abd,self.data,self.model,self.joint_ids,palm,self.thumb_links,R, hand_side)
            case "annular":
                move_annular(dr,self.data,self.model,self.joint_ids,palm,self.annular_links,R,hand)

    def get_contact_force(self, hand_id: int, finger: str) -> float:
        sensor_name = "left" if hand_id == 0 else "right"
        sensor_name += "_fingertip_" + finger
        # e.g. left_fingertip_thumb
        data = self.data.sensor(sensor_name).data
        if max(data) > data[0]:
            print("MAX:", max(data))
        if data[0] > self.maxForce:
            self.maxForce = data[0]
            print(self.maxForce)
        # data is an array containing only one number: the normal force
        return data[0] / 200

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

# def quaternion_multiply(quaternion1, quaternion0):
#     w0, x0, y0, z0 = quaternion0
#     w1, x1, y1, z1 = quaternion1
#     return [-x1*x0 - y1*y0 - z1*z0 + w1*w0,
#                         x1*w0 + y1*z0 - z1*y0 + w1*x0,
#                         -x1*z0 + y1*w0 + z1*x0 + w1*y0,
#                         x1*y0 - y1*x0 + z1*w0 + w1*z0]