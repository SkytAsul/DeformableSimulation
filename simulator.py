from engine import Engine
from coppelia import CoppeliaConnector
from mujoco_connector import MujocoConnector
from weart import WeartConnector
from benchmarking import Benchmarker, Plotter
from threading import Thread
from pynput import keyboard
import math

def closure_to_angle(closure):
    # dumb calc: between 0 and 0.4 we are proportional to 0 and 40°
    degree = closure * 100 if closure < 0.4 else 40
    return math.radians(degree)

def simulation(engine: Engine, weart: WeartConnector, openxr):
    print("Starting simulation.")
    engine.start_simulation()
    weart.start_listeners()

    perf_bench = Benchmarker(title="Performance profiler")
    force_plot = Plotter(title="Applied force graph")

    esc_pressed = False
    def key_press(key):
        nonlocal esc_pressed
        if key == keyboard.Key.esc:
            esc_pressed = True

    def loop():
        listener = keyboard.Listener(on_press=key_press)
        listener.start()

        try:
            while not esc_pressed:
                perf_bench.new_iteration()
                force_plot.new_iteration()
                angle = closure_to_angle(weart.get_index_closure())
                perf_bench.mark("Closure angle computation")
                engine.move_finger(angle)
                perf_bench.mark("Apply finger rotation")
                engine.step_simulation()
                perf_bench.mark("Do simulation step")
                force = engine.get_contact_force()
                perf_bench.mark("Get contact force")
                weart.apply_force(force)
                perf_bench.mark("Apply force to finger")
                force_plot.plot(force, "Force applied")
                force_plot.end_iteration()
                perf_bench.end_iteration()
        except KeyboardInterrupt:
            pass
        finally:
            force_plot.stop()
            perf_bench.stop()
            #perf_bench.export_csv("benchmark.csv", include_time=True)
            print("Stopping simulation...")
            engine.stop_simulation()
            listener.stop()

    t = Thread(target=loop)
    t.start()
    # we must run the loop in another thread because the graph can only be visualized in the main thread...
    #perf_bench.graph_viz(max_points=80, use_time=True)
    #force_plot.graph_viz(max_points=80, y_axis="Force")
    t.join()

if __name__ == "__main__":
    print("Starting script...\n")

    # print("Connecting to Coppelia...")
    # copp = CoppeliaConnector()
    # print("Connected.\n")

    print("Loading MuJoCo...")
    mujoco = MujocoConnector("assets/MuJoCo scene.xml")
    print("Loaded.")

    engine = mujoco

    print("Connecting to WEART...")
    with WeartConnector(ip_address="vps", port=10000) as weart:
        print("Connected. Calibrating...")

        weart.calibrate()
        print("Calibrated.\n")

        # openxr code
        simulation(engine, weart, None)