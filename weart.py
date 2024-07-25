from weartsdk import *
from weartsdk.WeArtCommon import HandSide, ActuationPoint
import time
import logging
from dataclasses import dataclass

HAPTIC_FINGERS = ["thumb", "index", "middle"]

@dataclass
class Finger:
    hand: int
    finger: str
    haptic_object: WeArtHapticObject
    touch_effect: TouchEffect
    thimble_tracking: WeArtThimbleTrackingObject

class WeartConnector(object):
    def __init__(self, enabled_hands: list[int], ip_address = WeArtCommon.DEFAULT_IP_ADDRESS, port = WeArtCommon.DEFAULT_TCP_PORT):
        self._client = WeArtClient(ip_address, port, log_level=logging.INFO)

        self._fingers: dict[str, Finger] = {}
        for hand in enabled_hands:
            hand_side = HandSide.Left.value if hand == 0 else HandSide.Right.value
            for finger in HAPTIC_FINGERS:
                match finger:
                    case "thumb":
                        actuation_point = ActuationPoint.Thumb
                    case "index":
                        actuation_point = ActuationPoint.Index
                    case "middle":
                        actuation_point = ActuationPoint.Middle

                haptic_object = WeArtHapticObject(self._client)
                haptic_object.handSideFlag = hand_side
                haptic_object.actuationPointFlag = actuation_point

                touch_effect = TouchEffect(WeArtTemperature(), WeArtForce(), WeArtTexture())
                haptic_object.AddEffect(touch_effect)

                thimble_tracking = WeArtThimbleTrackingObject(hand_side, actuation_point)

                self._fingers[WeartConnector.get_finger_id(hand, finger)] = Finger(hand, finger, haptic_object, touch_effect, thimble_tracking)
    
    def __enter__(self):
        self._client.Run()
        self._client.Start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._client.StopRawData()
        self._client.Stop()
        time.sleep(.5)
        self._client.Close()
    
    def calibrate(self):
        calibration = WeArtTrackingCalibration()
        self._client.AddMessageListener(calibration)
        self._client.StartCalibration()

        while(not calibration.getResult()):
            time.sleep(1)
        
        self._client.StopCalibration()
        self._client.RemoveMessageListener(calibration)
    
    def start_listeners(self):
        for finger in self._fingers.values():
            self._client.AddThimbleTracking(finger.thimble_tracking)

    def get_index_closure(self, hand_id: int, finger: str):
        thimble_tracking = self._fingers[WeartConnector.get_finger_id(hand_id, finger)].thimble_tracking
        return thimble_tracking.GetClosure()
    
    def apply_force(self, hand_id: int, finger: str, force_value: float):
        finger: Finger = self._fingers.get(WeartConnector.get_finger_id(hand_id, finger))

        finger.touch_effect.Set(finger.touch_effect.getTemperature(), WeArtForce(True, force_value), finger.touch_effect.getTexture())
        finger.haptic_object.UpdateEffects()

    @staticmethod
    def get_finger_id(hand_id: int, finger: str):
        return f"{finger}{hand_id}"