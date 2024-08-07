from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from enum import Enum
from interfaces import Engine
import math

class SceneException(Exception):
    pass

class CoppeliaConnector(Engine):
    def __init__(self):
        self._client = RemoteAPIClient()
        self.sim = self._client.require('sim')

        self._check_correct_engine()
        #self._mujoco = self._client.require('simMujoco')

        self._fetch_finger()

        self._force_calculator = SensorForceCalculator(self)

    def _check_correct_engine(self):
        engine = self.sim.getInt32Param(self.sim.intparam_dynamic_engine)
        if engine != self.sim.physics_mujoco:
            raise SceneException("The scene must run using the MuJoCo physics engine.")

    def _fetch_finger(self):
        self.finger_handle = self.sim.getObject("/Finger")
        self._finger_base_pos = self.sim.getObjectPose(self.finger_handle)
        _, _, finger_dim = self.sim.getShapeGeomInfo(self.finger_handle)
        self._finger_length = finger_dim[2]

    def move_finger(self, hand_id: int, finger: str, closure: float):
        if hand_id == 1 and finger == "index":
            angle = closure * 100 if closure < 0.4 else 40
            new_pos = self.sim.rotateAroundAxis(self._finger_base_pos, (1, 0, 0), (self._finger_base_pos[0], self._finger_base_pos[1] + self._finger_length, self._finger_base_pos[2]), angle)
            self.sim.setObjectPose(self.finger_handle, new_pos)

    def get_contact_force(self, hand_id: int, finger: str):
        return self._force_calculator.compute()
    
    def start_simulation(self):
        self.sim.setStepping(True)
        self.sim.startSimulation()
    
    def step_simulation(self, duration: float | None):
        self.sim.step()
    
    def stop_simulation(self):
        self.sim.stopSimulation()

class ForceCalculator:
    def __init__(self, copp: CoppeliaConnector):
        self._copp = copp

    def compute(self):
        """
        Computes the force received by the finger at any given moment.

        Should return a value between 0 and 1, 0 being "no force" and 1 being "maximum force". 
        """
        raise NotImplementedError

class ContactForceCalculation(Enum):
    SUM = 1,
    DOT_NORMAL = 2,
class ContactForceCalculator(ForceCalculator):
    def __init__(self, copp, calculation = ContactForceCalculation.DOT_NORMAL, max_iterations = 1):
        super().__init__(copp)
        self._calculation = calculation
        self._max_iterations = max_iterations # defaults to 1 seems like we get a big value first and then small ones
    
    def compute(self):
        total_force = 0
        for i in range(0, self._max_iterations):
            _, _, rForce, n = self._copp.sim.getContactInfo(self._copp.sim.handle_all, self._copp.finger_handle, i)
            if rForce == []:
                break # no more contacts

            match self._calculation:
                case ContactForceCalculation.SUM:
                    # easy way: sum all components
                    for j in range(0, 3):
                        total_force += rForce[j]
                case ContactForceCalculation.DOT_NORMAL:
                    # dot product of the force with the normal
                    for j in range(0, 3):
                        total_force += rForce[j] * n[j]
        
        total_force = abs(total_force) / 3
        # We divide by 3 because the values fetched this way seem to be between 0 and 3 so we scale down.
        return total_force

class SensorForceCalculator(ForceCalculator):
    MAX_FORCE = 6

    def __init__(self, copp):
        super().__init__(copp)
        self._fetch_force_sensor()

    def _fetch_force_sensor(self):
        self._force_sensor_handle = self._copp.sim.getObject("/FingertipForceSensor")
    
    def compute(self):
        result, forceVector, torqueVector = self._copp.sim.readForceSensor(self._force_sensor_handle)
        if result == 0:
            # no data available
            return 0
        total_force = forceVector[2] # the z component contains the force
        return abs(total_force) / SensorForceCalculator.MAX_FORCE
