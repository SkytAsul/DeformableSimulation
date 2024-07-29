from interfaces import Engine, Visualizer
from hand import Hand
from math import radians

import mujoco as mj
import mujoco.viewer as mj_viewer

import numpy as np
from middle_sim import *

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
        #self.L1y_middle = 0.015128056767373 #abs(L12[1])
        self.L1y_middle = np.linalg.norm([L12[1],L12[0]])
        #self.L1z_middle =  0.009446231805489869 #abs(L12[2])
        self.L1z_middle =  np.linalg.norm([L12[2],L12[0]])

        L23 = self.data.xpos[self.model.body('Middle_J3.stl').id] - self.data.xpos[self.model.body('Middle_J2.stl').id]
        #self.L2y_middle = 0.011112119117846375 #abs(L23[1])
        self.L2y_middle = np.linalg.norm([L23[1],L23[0]])
        #self.L2z_middle = 0.006617639381811979 #abs(L23[2])
        self.L2z_middle = np.linalg.norm([L23[2],L23[0]])

        L3T = self.data.site_xpos[self.model.site('forSensorMiddle_4.stl').id] - self.data.xpos[self.model.body('Middle_J3.stl').id]
        #self.L3y_middle = 0.004488227688179469 #abs(L3T[1])
        self.L3y_middle = np.linalg.norm([L3T[1],L3T[0]])
        #self.L3z_middle = 0.003247651090429904 #abs(L3T[2])
        self.L3z_middle = np.linalg.norm([L3T[2],L3T[0]])

        print(L12, L23, L3T, self.data.xpos[self.model.body('Middle_J2.stl').id], self.data.xpos[self.model.body('Middle_J1.stl').id])

        # Annular
        #L12 = self.data.xpos[self.model.body('Annular_J2.stl').id] - self.data.xpos[self.model.body('Annular_J1.stl').id]
        self.L1y_annular = 0.01578420772750644 #abs(L12[1])
        self.L1z_annular = 0.010347823357692132 #abs(L12[2])

        #L23 = self.data.xpos[self.model.body('Annular_J3.stl').id] - self.data.xpos[self.model.body('Annular_J2.stl').id]
        self.L2y_annular = 0.010817766376324478 #abs(L23[1])
        self.L2z_annular = 0.006137002359780264 #abs(L23[2])

        #L3T = self.data.site_xpos[self.model.site('forSensorAnnular_4.stl').id] - self.data.xpos[self.model.body('Annular_J3.stl').id]
        self.L3y_annular = 0.004465349215647008 #abs(L3T[1])
        self.L3z_annular = 0.003141688983307689 #abs(L3T[2])

        # Pinky
        #L12 = self.data.xpos[self.model.body('Pinky_J2.stl').id] - self.data.xpos[self.model.body('Pinky_J1.stl').id]
        self.L1y_pinky = 0.01578420772750644 #abs(L12[1])
        self.L1z_pinky =  0.010347823357692132 #abs(L12[2])

        #L23 = self.data.xpos[self.model.body('Pinky_J3.stl').id] - self.data.xpos[self.model.body('Pinky_J2.stl').id]
        self.L2y_pinky = 0.010817766376324478 #abs(L23[1])
        self.L2z_pinky = 0.006137002359780264 #abs(L23[2])

        #L3T = self.data.site_xpos[self.model.site('forSensorPinky_4.stl').id] - self.data.xpos[self.model.body('Pinky_J3.stl').id]
        self.L3y_pinky = 0.004465349215647008 #abs(L3T[1])
        self.L3z_pinky = 0.003141688983307689 #abs(L3T[2])
        
        # Index
        L12 = self.data.xpos[self.model.body('Index_J2.stl').id] - self.data.xpos[self.model.body('Index_J1.stl').id]
        #self.L1y_index = 0.01578420772750644 #abs(L12[1])
        #self.L1z_index = 0.010347823357692132 #abs(L12[2])
        self.L1y_index = np.linalg.norm([L12[1],L12[0]]) 
        self.L1z_index = np.linalg.norm([L12[2],L12[0]])

        L23 = self.data.xpos[self.model.body('Index_J3.stl').id] - self.data.xpos[self.model.body('Index_J2.stl').id]
        #self.L2y_index = 0.010817766376324478 #abs(L23[1])
        #self.L2z_index = 0.006137002359780264 #abs(L23[2])
        self.L2y_index = np.linalg.norm([L23[1],L23[0]])
        self.L2z_index = np.linalg.norm([L23[2],L23[0]])

        L3T = self.data.site_xpos[self.model.site('forSensor').id] - self.data.xpos[self.model.body('Index_J3.stl').id]
        #self.L3y_index = 0.004465349215647008 #abs(L3T[1])
        #self.L3z_index = 0.003141688983307689 #abs(L3T[2])
        self.L3y_index = np.linalg.norm([L3T[1],L3T[0]])
        self.L3z_index = np.linalg.norm([L3T[2],L3T[0]])
        

        # Thumb
        #L23 = self.data.xpos[self.model.body('Thumb_J3.stl').id] - self.data.xpos[self.model.body('Thumb_J2.stl').id]
        self.L2x_thumb = 0.010817766376324478 #abs(L23[0])
        self.L2y_thumb = 0.006137002359780264 #abs(L23[1])

        #L3T = self.data.site_xpos[self.model.site('forSensorThumb_3.stl').id] - self.data.xpos[self.model.body('Thumb_J3.stl').id]
        self.L3x_thumb = 0.004465349215647008 #abs(L3T[0])
        self.L3y_thumb = 0.003141688983307689 #abs(L3T[1])

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
                max_dist = 0.1
                min_dist = 0.02
            case "middle": 
                max_dist = 0.125
                min_dist = 0.0265
            case "thumb":
                max_dist = 0.1
                min_dist = 0
            case "annular": 
                max_dist = 0.08
                min_dist = 0.02
            case "pinky": 
                max_dist = 0.07
                min_dist = 0.02
        distance = max_dist - closure*(max_dist - min_dist)
        return distance

    def actuation(self, finger, distance):
        palm = self.data.xpos[self.model.body('right_hand_mocap').id]
        R = self.data.xmat[self.model.body('right_hand_mocap').id]
        R = np.reshape(R, [3, 3])

        match finger:
            case "middle":
                q1 = self.data.qpos[self.joint_ids['Middle_J1']]
                q2 = self.data.qpos[self.joint_ids['Middle_J2']]
                q3 = self.data.qpos[self.joint_ids['Middle_J3']]
                
                #end_effector_middle = self.model.site('forSensorMiddle_4.stl').id #"End-effector we wish to control.
                current_pose_middle = self.data.site_xpos[self.model.site('forSensorMiddle_4.stl').id] #Current pose


                offset_middle =  self.data.xpos[self.model.body('Middle_J1.stl').id] 
                offx_middle = offset_middle[0] - palm[0]
                offy_middle = offset_middle[1] - palm[1] 
                offz_middle = offset_middle[2] - palm[2] + 0.01

                #print(offset_middle, current_pose_middle)
                
                #lenght links [offsety, offsetz, l1y, l1z, l2y, l2z, l3y, l3z]
                links = [offx_middle, offy_middle, offz_middle, self.L1y_middle, self.L1z_middle, self.L2y_middle, self.L2z_middle, self.L3y_middle, self.L3z_middle]
                
                qdot, self.integral_middle = actuation_middle(distance, [q1, q2, q3], current_pose_middle, links, self.integral_middle, palm, R)

                v1 = qdot[0][0] 
                v2 = qdot[1][0] 
                v3 = qdot[2][0] 
                #v1 = 15
                #v2 = 15
                #v3 = 15
                
                # for i in range(len(self.data.qvel)):
                #     self.data.qvel[i] = 0
                # for i in range(6):
                #     self.data.qpos[i] = 0.1

                
                self.data.qvel[self.joint_ids['Middle_J1']] = v1
                self.data.qvel[self.joint_ids['Middle_J2']] = v2
                self.data.qvel[self.joint_ids['Middle_J3']] = v3
                # print("qvel:" , v1, v2, v3)

            case "index":
                q1 = self.data.qpos[self.joint_ids['Index_J1']]
                q2 = self.data.qpos[self.joint_ids['Index_J2']]
                q3 = self.data.qpos[self.joint_ids['Index_J3']]

                #end_effector_id = self.model.site('forSensor').id #"End-effector we wish to control.
                current_pose_index = self.data.site_xpos[self.model.site('forSensor').id] #Current pose

                #print(np.linalg.norm(current_pose_index - np.array([0, 0, 1])))

                offset_index =  self.data.xpos[self.model.body('Index_J1.stl').id] 
                offx_index = offset_index[0] - palm[0]
                offy_index = offset_index[1] - palm[1]
                offz_index = offset_index[2] - palm[2] + 0.01
                #print((current_pose_index - self.data.xpos[self.model.body('Index_J1.stl').id])[0])
                #lenght links [offsety, offsetz, l1y, l1z, l2y, l2z, l3y, l3z]
                links = [offx_index, offy_index, offz_index, self.L1y_index, self.L1z_index, self.L2y_index, self.L2z_index, self.L3y_index, self.L3z_index]

                qdot, self.integral_index = actuation_index(distance, [q1, q2, q3], current_pose_index, links, self.integral_index, palm,R)

                v1 = qdot[0][0] 
                v2 = qdot[1][0] 
                v3 = qdot[2][0] 
                
                # v1 = 0
                # v2 = 0
                # v3 = -0.1
            
                self.data.qvel[self.model.joint('Index_J1.stl').id] = v1
                self.data.qvel[self.model.joint('Index_J2.stl').id] = v2
                self.data.qvel[self.model.joint('Index_J3.stl').id] = v3
                
               
                
                print(self.data.ctrl[self.model.actuator('Index_J1.stl').id],
                      self.data.ctrl[self.model.actuator('Index_J2.stl').id],
                      self.data.ctrl[self.model.actuator('Index_J3.stl').id])

            case "thumb":
                q1 = self.data.qpos[self.joint_ids['Thumb_J2']]
                q2 = self.data.qpos[self.joint_ids['Thumb_J3']]

                offset_thumb =  self.data.xpos[self.model.body('Thumb_J2.stl').id] 
                offx_thumb = offset_thumb[0] - palm[0] + 0.035
                offy_thumb = offset_thumb[1] - palm[1] + 0.02

                #end_effector_id = self.model.site('forSensorThumb_3.stl').id #"End-effector we wish to control.
                current_pose_thumb = self.data.site_xpos[self.model.site('forSensorThumb_3.stl').id] #Current pose

                #print("thumb: ", current_pose_thumb)
                #print(np.linalg.norm(current_pose_thumb - np.array([0, 0, 1])))

                links = [offx_thumb, offy_thumb, offset_thumb[2], self.L2x_thumb, self.L2y_thumb, self.L3x_thumb, self.L3y_thumb]

                qdot, self.integral_thumb = actuation_thumb(distance, [q1, q2], current_pose_thumb, links, self.integral_thumb, palm)

                v1 = qdot[0][0]
                v2 = qdot[1][0]

                self.data.qvel[self.joint_ids['Thumb_J2']] = v1
                self.data.qvel[self.joint_ids['Thumb_J3']] = v2

            case "annular":
                q1 = self.data.qpos[self.joint_ids['Annular_J1']]
                q2 = self.data.qpos[self.joint_ids['Annular_J2']]
                q3 = self.data.qpos[self.joint_ids['Annular_J3']]
                
                current_pose_annular = self.data.site_xpos[self.model.site('forSensorAnnular_4.stl').id] #Current pose

                offset_annular =  self.data.xpos[self.model.body('Annular_J1.stl').id] 
                offy_annular = offset_annular[1] - palm[1] + 0.03
                offz_annular = offset_annular[2] - palm[2] + 0.01

                #lenght links [offsety, offsetz, l1y, l1z, l2y, l2z, l3y, l3z]
                links = [offy_annular, offz_annular, self.L1y_annular, self.L1z_annular, self.L2y_annular, self.L2z_annular, self.L3y_annular, self.L3z_annular]
                
                qdot, self.integral_annular = actuation_annular(distance, [q1, q2, q3], current_pose_annular, links, self.integral_annular, palm)

                #print("Annular: ", current_pose_annular)
                #print("Annular: ", np.linalg.norm(current_pose_annular - np.array([0, 0, 1])))
                
                v1 = qdot[0][0] 
                v2 = qdot[1][0] 
                v3 = qdot[2][0] 

                self.data.qvel[self.joint_ids['Annular_J1']] = v1
                self.data.qvel[self.joint_ids['Annular_J2']] = v2
                self.data.qvel[self.joint_ids['Annular_J3']] = v3
            
            case "pinky":
                q1 = self.data.qpos[self.joint_ids['Pinky_J1']]
                q2 = self.data.qpos[self.joint_ids['Pinky_J2']]
                q3 = self.data.qpos[self.joint_ids['Pinky_J3']]
                
                current_pose_pinky = self.data.site_xpos[self.model.site('forSensorPinky_4.stl').id] #Current pose

                offset_pinky =  self.data.xpos[self.model.body('Pinky_J1.stl').id] 
                offy_pinky = offset_pinky[1] - palm[1] 
                offz_pinky = offset_pinky[2] - palm[2] 

                #print("Pinky: ", np.linalg.norm(current_pose_pinky - np.array([0, 0, 1])))
                
                #lenght links [offsety, offsetz, l1y, l1z, l2y, l2z, l3y, l3z]
                links = [offy_pinky, offz_pinky, self.L1y_pinky, self.L1z_pinky, self.L2y_pinky, self.L2z_pinky, self.L3y_pinky, self.L3z_pinky]
                
                qdot, self.integral_pinky = actuation_pinky(distance, [q1, q2, q3], current_pose_pinky, links, self.integral_pinky, palm)

                v1 = qdot[0][0] 
                v2 = qdot[1][0] 
                v3 = qdot[2][0] 

                self.data.qvel[self.joint_ids['Pinky_J1']] = v1
                self.data.qvel[self.joint_ids['Pinky_J2']] = v2
                self.data.qvel[self.joint_ids['Pinky_J3']] = v3

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