from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math

class SceneException(Exception):
    pass

class CoppeliaConnector:
    def __init__(self):
        self._client = RemoteAPIClient()
        self._sim = self._client.require('sim')

        self._check_correct_engine()
        #self._mujoco = self._client.require('simMujoco')

        self._fetch_finger()

    def _check_correct_engine(self):
        engine = self._sim.getInt32Param(self._sim.intparam_dynamic_engine)
        if engine != self._sim.physics_mujoco:
            raise SceneException("The scene must run using the MuJoCo physics engine.")

    def _fetch_finger(self):
        self._finger_handle = self._sim.getObject("/Finger")
        self._finger_base_pos = self._sim.getObjectPose(self._finger_handle)
        _, _, finger_dim = self._sim.getShapeGeomInfo(self._finger_handle)
        self._finger_length = finger_dim[2]

    def move_finger(self, angle):
        new_pos = self._sim.rotateAroundAxis(self._finger_base_pos, (1, 0, 0), (self._finger_base_pos[0], self._finger_base_pos[1] + self._finger_length, self._finger_base_pos[2]), angle)
        self._sim.setObjectPose(self._finger_handle, new_pos)

    def get_contact_force(self):
        max_iterations = 1 # seems like we get a big value first and then small ones
        total_force = 0
        for i in range(0, max_iterations):
            _, _, rForce, _ = self._sim.getContactInfo(self._sim.handle_all, self._finger_handle, i)
            if rForce == []:
                break
            for j in range(0, 3):
                total_force += rForce[j]
        return total_force
    
    def start_simulation(self):
        self._sim.setStepping(True)
        self._sim.startSimulation()
    
    def step_simulation(self):
        self._sim.step()
    
    def stop_simulation(self):
        self._sim.stopSimulation()