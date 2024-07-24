from interfaces import Engine, Visualizer
from pynput import keyboard
import mujoco as mj
import mujoco.viewer as mj_viewer
import math
from math import sin, cos, pi, sqrt
import numpy as np
from hand.thumb_sim import *
from hand.index_sim import *
from hand.middle_sim import *

class MujocoConnector(Engine):
    def __init__(self, xml_path : str):
        """Creates the MuJoCo Connector with the MJCF at the passed path.

        Args:
            xml_path (str): path to the XML file containing the MJCF
        """
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self._fetch_finger_joints()
        self._fetch_hand()
        # self.model.opt.disableflags = 32

    def _fetch_finger_joints(self):
        self.joint_ids = {
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
        self.offset_middle =  self.data.xpos[self.model.body('Middle_J1.stl').id]

        L12 = self.data.xpos[self.model.body('Middle_J2.stl').id] - self.data.xpos[self.model.body('Middle_J1.stl').id]
        self.L1y_middle = abs(L12[1])
        self.L1z_middle =  abs(L12[2])

        L23 = self.data.xpos[self.model.body('Middle_J3.stl').id] - self.data.xpos[self.model.body('Middle_J2.stl').id]
        self.L2y_middle = abs(L23[1])
        self.L2z_middle = abs(L23[2])

        L3T = self.data.site_xpos[self.model.site('forSensorMiddle_4.stl').id] - self.data.xpos[self.model.body('Middle_J3.stl').id]
        self.L3y_middle = abs(L3T[1])
        self.L3z_middle = abs(L3T[2])
        
        # Index
        self.offset_index =  self.data.xpos[self.model.body('Index_J1.stl').id]

        L12 = self.data.xpos[self.model.body('Index_J2.stl').id] - self.data.xpos[self.model.body('Index_J1.stl').id]
        self.L1y_index = abs(L12[1])
        self.L1z_index = abs(L12[2])

        L23 = self.data.xpos[self.model.body('Index_J3.stl').id] - self.data.xpos[self.model.body('Index_J2.stl').id]
        self.L2y_index = abs(L23[1])
        self.L2z_index = abs(L23[2])

        L3T = self.data.site_xpos[self.model.site('forSensor').id] - self.data.xpos[self.model.body('Index_J3.stl').id]
        self.L3y_index = abs(L3T[1])
        self.L3z_index = abs(L3T[2])

        # Thumb
        self.offset_thumb =  self.data.xpos[self.model.body('Thumb_J2.stl').id]

        L23 = self.data.xpos[self.model.body('Thumb_J3.stl').id] - self.data.xpos[self.model.body('Thumb_J2.stl').id]
        self.L2x_thumb = abs(L23[0])
        self.L2y_thumb = abs(L23[1])

        L3T = self.data.site_xpos[self.model.site('forSensorThumb_3.stl').id] - self.data.xpos[self.model.body('Thumb_J3.stl').id]
        self.L3x_thumb = abs(L3T[0])
        self.L3y_thumb = abs(L3T[1])

        self.timestep = self.model.opt.timestep

        self.integral_middle = 0
        self.integral_index = 0
        self.integral_thumb = 0

    def mapping(self, closure, finger):
        max_dist = 0
        min_dist = 0
        match finger:
            case "index": 
                max_dist = 0.75
                min_dist = 0.2
            case "middle": 
                max_dist = 1
                min_dist = 0.2
            case "thumb":
                max_dist = 0.8
                min_dist = 0
        distance = max_dist - closure*(max_dist - min_dist)
        return distance

    def actuation(self, finger, distance):
        match finger:
            case "middle":
                # MIDDLE
                q1 = self.data.qpos[self.joint_ids['Middle_J1']]
                q2 = self.data.qpos[self.joint_ids['Middle_J2']]
                q3 = self.data.qpos[self.joint_ids['Middle_J3']]
                
                #end_effector_middle = self.model.site('forSensorMiddle_4.stl').id #"End-effector we wish to control.
                current_pose_middle = self.data.site_xpos[self.model.site('forSensorMiddle_4.stl').id] #Current pose

                #lenght links [offsety, offsetz, l1y, l1z, l2y, l2z, l3y, l3z]
                links = [self.offset_middle[1], self.offset_middle[2], self.L1y_middle, self.L1z_middle, self.L2y_middle, self.L2z_middle, self.L3y_middle, self.L3z_middle]
                
                qdot, self.integral_middle, d_middle = actuation_middle(distance, [q1, q2, q3], current_pose_middle, links, self.integral_middle, self.timestep)

                v1 = qdot[0][0] 
                v2 = qdot[1][0] 
                v3 = qdot[2][0] 

                self.data.qvel[self.joint_ids['Middle_J1']] = v1
                self.data.qvel[self.joint_ids['Middle_J2']] = v2
                self.data.qvel[self.joint_ids['Middle_J3']] = v3

                # Movements of anular and pinky
                self.data.qvel[self.model.joint('Annular_J1.stl').id] = 0.9*v1
                self.data.qvel[self.model.joint('Annular_J2.stl').id] = 0.95*v2
                self.data.qvel[self.model.joint('Annular_J3.stl').id] = 0.9*v3

                self.data.qvel[self.model.joint('Pinky_J1.stl').id] = 1*v1
                self.data.qvel[self.model.joint('Pinky_J2.stl').id] = 0.85*v2
                self.data.qvel[self.model.joint('Pinky_J3.stl').id] = 0.8*v3

            case "index":
                # INDEX
                q1 = self.data.qpos[self.joint_ids['Index_J1']]
                q2 = self.data.qpos[self.joint_ids['Index_J2']]
                q3 = self.data.qpos[self.joint_ids['Index_J3']]

                #end_effector_id = self.model.site('forSensor').id #"End-effector we wish to control.
                current_pose_index = self.data.site_xpos[self.model.site('forSensor').id] #Current pose

                #print(np.linalg.norm(current_pose_index - np.array([0, 0, 1])))

                #lenght links [offsety, offsetz, l1y, l1z, l2y, l2z, l3y, l3z]
                links = [self.offset_index[1], self.offset_index[2], self.L1y_index, self.L1z_index, self.L2y_index, self.L2z_index, self.L3y_index, self.L3z_index]

                qdot, self.integral_index, d_index = actuation_index(distance, [q1, q2, q3], current_pose_index, links, self.integral_index, self.timestep)

                v1 = qdot[0][0] 
                v2 = qdot[1][0] 
                v3 = qdot[2][0] 

                self.data.qvel[self.joint_ids['Index_J1']] = v1
                self.data.qvel[self.joint_ids['Index_J2']] = v2
                self.data.qvel[self.joint_ids['Index_J3']] = v3

            case "thumb":
                # THUMB
                q1 = self.data.qpos[self.joint_ids['Thumb_J2']]
                q2 = self.data.qpos[self.joint_ids['Thumb_J3']]

                #end_effector_id = self.model.site('forSensorThumb_3.stl').id #"End-effector we wish to control.
                current_pose_thumb = self.data.site_xpos[self.model.site('forSensorThumb_3.stl').id] #Current pose

                #print("thumb: ", current_pose_thumb)
                #print(np.linalg.norm(current_pose_thumb - np.array([0, 0, 1])))

                links = [self.offset_thumb[0], self.offset_thumb[1], self.L2x_thumb, self.L2y_thumb, self.L3x_thumb, self.L3y_thumb]

                qdot, self.integral_thumb, d_thumb = actuation_thumb(distance, [q1, q2], current_pose_thumb, links, self.integral_thumb, self.timestep)

                v1 = qdot[0][0]
                v2 = qdot[1][0]

                self.data.qvel[self.joint_ids['Thumb_J2']] = v1
                self.data.qvel[self.joint_ids['Thumb_J3']] = v2

    def _fetch_hand(self):
        self._right_hand_id = self.model.body("right_hand_mocap").mocapid[0]
    
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

class MujocoSimpleVisualizer(Visualizer):
    def __init__(self, mujoco : MujocoConnector):
        self._mujoco = mujoco
        self._scene = mj.MjvScene(mujoco.model, 1000)
    
    def _key_press(self, key):
        if key == keyboard.Key.esc:
            self._esc_pressed = True
    
    def start_visualization(self):
        self._viewer = mj_viewer.launch_passive(self._mujoco.model, self._mujoco.data)
        self._viewer.cam.azimuth = 138
        self._viewer.cam.distance = 0.5
        self._viewer.cam.elevation = -16
        # self._viewer.cam.lookat = self._mujoco._finger_base_pos.copy()
        
        self._esc_pressed = False
        self._kb_listener = keyboard.Listener(on_press=self._key_press)
        self._kb_listener.start()

    def render_frame(self):
        self._viewer.sync()

    def should_exit(self):
        return self._esc_pressed or not self._viewer.is_running()
    
    def stop_visualization(self):
        self._kb_listener.stop()
        self._viewer.close()