from enum import Enum
from weartsdk import *
from weartsdk.WeArtCommon import HandSide, ActuationPoint, MiddlewareStatus
import time
import logging
from threading import Thread
from dataclasses import dataclass
from colorama import Fore, Style

HAPTIC_FINGERS = ["thumb", "index", "middle"]

@dataclass
class Finger:
    hand: int
    finger: str
    haptic_object: WeArtHapticObject
    touch_effect: TouchEffect
    thimble_tracking: WeArtThimbleTrackingObject

def error(*values: object):
    print(Fore.RED + Style.BRIGHT, "WEART:" + Style.NORMAL, *values, Fore.RESET)

def info(*values: object):
    print(Fore.YELLOW + Style.BRIGHT, "WEART:" + Style.NORMAL, *values, Fore.RESET)

class WeartConnector(object):
    def __init__(self, enabled_hands: list[int], ip_address = WeArtCommon.DEFAULT_IP_ADDRESS, port = WeArtCommon.DEFAULT_TCP_PORT):
        self._client = WeArtClient(ip_address, port, log_level=logging.INFO)

        self._fingers: dict[str, Finger] = {}
        for hand in enabled_hands:
            hand_side = HandSide.Left if hand == 0 else HandSide.Right
            for finger in HAPTIC_FINGERS:
                match finger:
                    case "thumb":
                        actuation_point = ActuationPoint.Thumb
                    case "index":
                        actuation_point = ActuationPoint.Index
                    case "middle":
                        actuation_point = ActuationPoint.Middle
                    case _:
                        raise RuntimeError()

                haptic_object = WeArtHapticObject(self._client)
                haptic_object.handSideFlag = hand_side.value
                haptic_object.actuationPointFlag = actuation_point

                touch_effect = TouchEffect(WeArtTemperature(), WeArtForce(), WeArtTexture())
                haptic_object.AddEffect(touch_effect)

                thimble_tracking = WeArtThimbleTrackingObject(hand_side, actuation_point)

                self._fingers[WeartConnector.get_finger_id(hand, finger)] = Finger(hand, finger, haptic_object, touch_effect, thimble_tracking)
    
    def __enter__(self):
        self._status_manager = WeartStatusManager(self)
        self._client.Run()
        self._status_manager.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._status_manager.stop()
        self._client.Close()
    
    def _start_finger_listeners(self):
        for finger in self._fingers.values():
            self._client.AddThimbleTracking(finger.thimble_tracking)
    
    def _stop_finger_listeners(self):
        for finger in self._fingers.values():
            self._client.RemoveMessageListener(finger.thimble_tracking)

    def get_index_closure(self, hand_id: int, finger: str):
        thimble_tracking = self._fingers[WeartConnector.get_finger_id(hand_id, finger)].thimble_tracking
        return thimble_tracking.GetClosure()
    
    def apply_force(self, hand_id: int, finger: str, force_value: float):
        finger_obj = self._fingers[WeartConnector.get_finger_id(hand_id, finger)]

        finger_obj.touch_effect.Set(finger_obj.touch_effect.getTemperature(), WeArtForce(True, force_value), finger_obj.touch_effect.getTexture())
        finger_obj.haptic_object.UpdateEffects()

    @staticmethod
    def get_finger_id(hand_id: int, finger: str):
        return f"{finger}{hand_id}"

class WeartSimulationStatus(Enum):
    NOT_STARTED = 0
    FULLY_STARTED = 2
    STOP_TO_RESTART = 3
    STOP_TO_EXIT = 4

class WeartStatusManager(Thread):
    """This class implements some ways to automatically recover from a WEART failure by restarting.
    """
    def __init__(self, connector: WeartConnector):
        super().__init__()

        self._connector = connector
        self._no_data = True

        self._listener = MiddlewareStatusListener()
        self._listener.AddStatusCallback(self._status_changed)
        self._connector._client.AddMessageListener(self._listener)

    def _status_changed(self, data: MiddlewareStatusUpdate):
        # print(data)
        self._no_data = False
        if data.statusCode != 0:
            error(f"{data.errorDesc} (error {data.statusCode})")

    def run(self):
        self._status = WeartSimulationStatus.NOT_STARTED

        while True:
            weart_status = self._listener.lastStatus().status

            # beware order of match cases!
            match self._status, weart_status:
                case WeartSimulationStatus.STOP_TO_EXIT, MiddlewareStatus.IDLE:
                    break # over!
                case WeartSimulationStatus.STOP_TO_EXIT, MiddlewareStatus.DISCONNECTED:
                    break # over!
                # TODO: investigate why using | to do one single case does not work
                case WeartSimulationStatus.STOP_TO_EXIT, _:
                    info("Waiting for stop...")
                case _, MiddlewareStatus.DISCONNECTED if not self._no_data:
                    # For some reason, when we connect to the SDK the status is DISCONNECTED by default,
                    # even if in the SDK it is IDLE or something else. Hence the guard clause.
                    error("USB dongle is not connected")
                case _, MiddlewareStatus.CALIBRATION:
                    # This loop should NEVER see calibration mode: when we ask for calibration,
                    # it is with this thread and we wait until calibration completion.
                    error("In calibration mode but should not.")
                case WeartSimulationStatus.NOT_STARTED, MiddlewareStatus.STARTING:
                    info("In starting...")
                case WeartSimulationStatus.NOT_STARTED, MiddlewareStatus.RUNNING:
                    # Means the Start() call followed in a successful start.
                    # We can initiate calibration.
                    self._calibrate()
                case WeartSimulationStatus.NOT_STARTED, _:
                    self._try_start()
                case WeartSimulationStatus.FULLY_STARTED, MiddlewareStatus.RUNNING:
                    pass # Everything is good!
                case WeartSimulationStatus.FULLY_STARTED, _:
                    self._stopped_working(weart_status)
                case WeartSimulationStatus.STOP_TO_RESTART, MiddlewareStatus.IDLE:
                    self._status = WeartSimulationStatus.NOT_STARTED
                    self._try_start()
                case WeartSimulationStatus.STOP_TO_RESTART, _:
                    info("Waiting before restart...")
            
            time.sleep(1)
        info("Stopped.")
    
    def stop(self):
        self._status = WeartSimulationStatus.STOP_TO_EXIT
        self._connector._client.Stop()
        self.join()
    
    def _try_start(self):
        if not self._no_data and len(self._listener.lastStatus().devices) == 0:
            error("No device connected.")
        else:
            info("Starting...")
            self._connector._client.Start()

    def _finished_starting(self):
        self._status = WeartSimulationStatus.FULLY_STARTED
        self._connector._start_finger_listeners()
    
    def _stopped_working(self, status: MiddlewareStatus):
        error(f"Stopped. Status: {status}. Trying to restart...")
        self._status = WeartSimulationStatus.STOP_TO_RESTART
        self._connector._client.Stop() # force stop to restart from the beginning

    def _calibrate(self):
        info("Starting calibration. Stand still...")

        calibration = WeArtTrackingCalibration()
        self._connector._client.AddMessageListener(calibration)
        self._connector._client.StartCalibration()

        while(not calibration.getResult()):
            time.sleep(1)
        
        self._connector._client.StopCalibration()
        self._connector._client.RemoveMessageListener(calibration)

        info("Calibration is done.")

        self._finished_starting()

    
