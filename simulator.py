from coppelia import *
from weart import *
import math, keyboard

def closure_to_angle(closure):
    # dumb calc: between 0 and 0.3 we are proportional to 0 and 30°
    degree = closure * 100 if closure < 0.4 else 40
    return math.radians(degree)

def force_copp_to_weart(copp_force):
    # absolute value, scaled from 0 to 4 -> 0 to 1
    force = abs(copp_force)
    force /= 3
    return force

def simulation(copp: CoppeliaConnector, weart: WeartConnector, openxr):
    print("Starting simulation.")
    copp.start_simulation()
    weart.start_listeners()

    try:
        while not keyboard.is_pressed("esc"):
            angle = closure_to_angle(weart.get_index_closure())
            copp.move_finger(angle)
            copp.step_simulation()
            force = force_copp_to_weart(copp.get_contact_force())
            weart.apply_force(force)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping simulation...")
        copp.stop_simulation()

if __name__ == "__main__":
    print("Starting script...\n")

    print("Connecting to Coppelia...")
    copp = CoppeliaConnector()
    print("Connected.\n")

    print("Connecting to WEART...")
    with WeartConnector() as weart:
        print("Connected. Calibrating...")

        weart.calibrate()
        print("Calibrated.\n")

        # openxr code
        simulation(copp, weart, None)