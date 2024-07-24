from weartsdk import *
from weartsdk.WeArtCommon import HandSide, ActuationPoint
import time
import logging

class WeartConnector(object):
    def __init__(self, ip_address = WeArtCommon.DEFAULT_IP_ADDRESS, port = WeArtCommon.DEFAULT_TCP_PORT):
        self._client = WeArtClient(ip_address, port, log_level=logging.INFO)

        # Haptic for the index
        self.hapticObjectIndex = WeArtHapticObject(self._client)
        self.hapticObjectIndex.handSideFlag = HandSide.Right.value
        self.hapticObjectIndex.actuationPointFlag = ActuationPoint.Index
        self.touchEffectIndex = TouchEffect(WeArtTemperature(), WeArtForce(), WeArtTexture())
        self.hapticObjectIndex.AddEffect(self.touchEffectIndex)

        # Haptic for the middle        
        self.hapticObjectMiddle = WeArtHapticObject(self._client)
        self.hapticObjectMiddle.handSideFlag = HandSide.Right.value
        self.hapticObjectMiddle.actuationPointFlag = ActuationPoint.Middle
        self.touchEffectMiddle = TouchEffect(WeArtTemperature(), WeArtForce(), WeArtTexture())
        self.hapticObjectMiddle.AddEffect(self.touchEffectMiddle)

        # Haptic for the thumb
        self.hapticObjectThumb = WeArtHapticObject(self._client)
        self.hapticObjectThumb.handSideFlag = HandSide.Right.value
        self.hapticObjectThumb.actuationPointFlag = ActuationPoint.Thumb
        self.touchEffectThumb = TouchEffect(WeArtTemperature(), WeArtForce(), WeArtTexture())
        self.hapticObjectThumb.AddEffect(self.touchEffectThumb)
    
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
    
    def start_listeners(self):
        self.indexThimbleTracking = WeArtThimbleTrackingObject(HandSide.Right, ActuationPoint.Index)
        self.middleThimbleTracking = WeArtThimbleTrackingObject(HandSide.Right, ActuationPoint.Middle)
        self.thumbThimbleTracking = WeArtThimbleTrackingObject(HandSide.Right, ActuationPoint.Thumb)
        self._client.AddThimbleTracking(self.indexThimbleTracking)
        self._client.AddThimbleTracking(self.middleThimbleTracking)
        self._client.AddThimbleTracking(self.thumbThimbleTracking)

    def get_index_closure(self):
        return self.indexThimbleTracking.GetClosure()
    
    def get_middle_closure(self):
        return self.middleThimbleTracking.GetClosure()
    
    
    def get_thumb_closure(self):
        return self.thumbThimbleTracking.GetClosure()
    
    
    def apply_force(self, force_value, finger):
        match finger:
            case 'index':
                self.touchEffectIndex.Set(self.touchEffectIndex.getTemperature(), WeArtForce(True, force_value), self.touchEffectIndex.getTexture())
                self.hapticObjectIndex.UpdateEffects()
            case 'middle':
                self.touchEffectMiddle.Set(self.touchEffectMiddle.getTemperature(), WeArtForce(True, force_value), self.touchEffectMiddle.getTexture())
                self.hapticObjectMiddle.UpdateEffects()
            case 'thumb':
                self.touchEffectThumb.Set(self.touchEffectThumb.getTemperature(), WeArtForce(True, force_value), self.touchEffectThumb.getTexture())
                self.hapticObjectThumb.UpdateEffects()