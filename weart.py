from weartsdk.WeArtHapticObject import WeArtHapticObject
from weartsdk.WeArtCommon import HandSide, ActuationPoint, CalibrationStatus
from weartsdk.WeArtTemperature import WeArtTemperature
from weartsdk.WeArtTexture import WeArtTexture
from weartsdk.WeArtForce import WeArtForce
from weartsdk.WeArtCommon import TextureType
from weartsdk.WeArtEffect import TouchEffect
from weartsdk.WeArtTrackingCalibration import WeArtTrackingCalibration
from weartsdk.WeArtThimbleTrackingObject import WeArtThimbleTrackingObject
from weartsdk.WeArtTrackingRawData import WeArtTrackingRawData
from weartsdk.MiddlewareStatusListener import MiddlewareStatusListener
from weartsdk.WeArtAnalogSensorData import WeArtAnalogSensorData

from weartsdk.WeArtClient import WeArtClient
import weartsdk.WeArtCommon
import time
import logging

class WeartConnector(object):
    def __init__(self):
        self._client = WeArtClient(weartsdk.WeArtCommon.DEFAULT_IP_ADDRESS, weartsdk.WeArtCommon.DEFAULT_TCP_PORT, log_level=logging.INFO)

        self._hapticObject = WeArtHapticObject(self._client)
        self._hapticObject.handSideFlag = HandSide.Right.value
        self._hapticObject.actuationPointFlag = ActuationPoint.Index
        self._touchEffect = TouchEffect(WeArtTemperature(), WeArtForce(), WeArtTexture())
        self._hapticObject.AddEffect(self._touchEffect)
    
    def __enter__(self):
        self._client.Run()
        self._client.Start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._client.StopRawData()
        self._client.Stop()
        self._client.Close()
    
    def calibrate(self):
        calibration = WeArtTrackingCalibration()
        self._client.AddMessageListener(calibration)
        self._client.StartCalibration()

        while(not calibration.getResult()):
            time.sleep(1)
        
        self._client.StopCalibration()
    
    '''
    def start_position(self):
        self._trackingRawSensorData = WeArtTrackingRawData(HandSide.Right, ActuationPoint.Index)
        self._client.AddMessageListener(self._trackingRawSensorData)
        self._client.StartRawData()
        while self._trackingRawSensorData.GetLastSample().timestamp == 0:
            time.sleep(1)
    '''
    
    def start_listeners(self):
        self._thumbThimbleTracking = WeArtThimbleTrackingObject(HandSide.Right, ActuationPoint.Index)
        self._client.AddThimbleTracking(self._thumbThimbleTracking)

    def get_index_closure(self):
        return self._thumbThimbleTracking.GetClosure()
    
    def apply_force(self, force_value):
        print("Applying", force_value)

        self._touchEffect.Set(self._touchEffect.getTemperature(), WeArtForce(True, force_value), self._touchEffect.getTexture())
        self._hapticObject.UpdateEffects()