from coppelia import *
from weart import *
import math, keyboard
from benchmarking import Benchmarker
from threading import Thread

def closure_to_angle(closure):
    # dumb calc: between 0 and 0.4 we are proportional to 0 and 40°
    degree = closure * 100 if closure < 0.4 else 40
    return math.radians(degree)

def force_copp_to_weart(copp_force):
    # absolute value, scaled from 0 to 3 -> 0 to 1
    force = abs(copp_force)
    force /= 3
    return force

def simulation(copp: CoppeliaConnector, weart: WeartConnector, openxr):
    print("Starting simulation.")
    copp.start_simulation()
    weart.start_listeners()

    bench = Benchmarker()

    def loop():
        try:
            while not keyboard.is_pressed("esc"):
                bench.new_iteration()
                angle = closure_to_angle(weart.get_index_closure())
                bench.mark("Closure angle computation")
                copp.move_finger(angle)
                bench.mark("Apply finger rotation")
                copp.step_simulation()
                bench.mark("Do simulation step")
                force = force_copp_to_weart(copp.get_contact_force())
                bench.mark("Get contact force")
                weart.apply_force(force)
                bench.mark("Apply force to finger")
                bench.end_iteration()
        except KeyboardInterrupt:
            pass
        finally:
            bench.stop()
            bench.export_csv("benchmark.csv", include_time=True)
            print("Stopping simulation...")
            copp.stop_simulation()

    t = Thread(target=loop)
    t.start()
    # we must run the loop in another thread because the graph can only be visualized in the main thread...
    bench.graph_viz(max_points=50, use_time=True)

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