# local code
from engine import Engine
from coppelia import CoppeliaConnector
from mujoco_connector import MujocoConnector, MujocoRenderer
from weart import WeartConnector
from benchmarking import Benchmarker, Plotter
from openxr import OpenXrConnector

# libraries
from typing import Optional
from threading import Thread
from pynput import keyboard
import math

def closure_to_angle(closure):
    # dumb calc: between 0 and 0.4 we are proportional to 0 and 40°
    degree = closure * 100 if closure < 0.4 else 40
    return math.radians(degree)

def simulation(engine: Engine, weart: WeartConnector, openxr: OpenXrConnector, renderer : Optional[MujocoRenderer]):
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
        
        if renderer is not None:
            renderer.init_context()

        try:
            for frame in openxr.main_loop():
                if esc_pressed:
                    break
                perf_bench.new_iteration()
                force_plot.new_iteration()
                angle = closure_to_angle(weart.get_index_closure())
                perf_bench.mark("Closure angle computation")
                if renderer is not None:
                    renderer.update_eyes()
                    perf_bench.mark("Apply eye positions")
                engine.move_finger(angle)
                perf_bench.mark("Apply finger rotation")
                engine.step_simulation()
                perf_bench.mark("Do simulation step")
                if renderer is not None:
                    renderer.render_scene()
                    perf_bench.mark("Render scene")
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

    threaded = False
    if threaded:
        t = Thread(target=loop)
        t.start()
        # we must run the loop in another thread because the graph can only be visualized in the main thread...
        #perf_bench.graph_viz(max_points=80, use_time=True)
        #force_plot.graph_viz(max_points=10000, y_axis="Force")
        t.join()
    else:
        loop()

if __name__ == "__main__":
    print("Starting script...\n")

    # print("Connecting to Coppelia...")
    # copp = CoppeliaConnector()
    # print("Connected.\n")

    print("Loading MuJoCo...")
    mujoco = MujocoConnector("assets/MuJoCo scene.xml", viewer=False)
    print("Loaded.")

    engine = mujoco

    print("Loading Virtual Reality...")
    with OpenXrConnector() as openxr:
        print("OpenXr context created.")

        renderer = MujocoRenderer(mujoco, openxr)

        print("Connecting to WEART...")
        with WeartConnector() as weart:
            print("Connected. Calibrating...")

            weart.calibrate()
            print("Calibrated.\n")

            simulation(engine, weart, openxr, renderer)