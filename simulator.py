# local code
from interfaces import *
from coppelia import CoppeliaConnector
from mujoco_connector import MujocoConnector, MujocoSimpleVisualizer
from mujoco_xr import MujocoXRVisualizer
from weart import WeartConnector
from benchmarking import Benchmarker, Plotter

# libraries
from threading import Thread
import math

def closure_to_angle(closure):
    # dumb calc: between 0 and 0.4 we are proportional to 0 and 40°
    degree = closure * 100 if closure < 0.4 else 40
    return math.radians(degree)

def simulation(engine: Engine,
                weart: WeartConnector | None,
                visualizer: Visualizer,
                hand: HandPoseProvider | None):
    print("Starting simulation.")
    engine.start_simulation()
    if weart is not None:
        weart.start_listeners()

    perf_bench = Benchmarker(title="Performance profiler")
    force_plot = Plotter(title="Applied force graph")

    def loop():
        visualizer.start_visualization()

        try:
            while not visualizer.should_exit():
                frame_continue = visualizer.start_frame()
                if visualizer.should_exit():
                    break
                if not frame_continue:
                    continue
                perf_bench.new_iteration()
                force_plot.new_iteration()

                if weart is not None:
                    angle = closure_to_angle(weart.get_index_closure())
                    perf_bench.mark("Closure angle computation")

                    engine.move_finger(angle)
                    perf_bench.mark("Apply finger rotation")

                engine.step_simulation()
                perf_bench.mark("Do simulation step")

                visualizer.render_frame()
                perf_bench.mark("Render")

                force = engine.get_contact_force()
                perf_bench.mark("Get contact force")

                if weart is not None:
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

            if visualizer is not None:
                print("Stopping visualization...")
                visualizer.stop_visualization()

            print("Stopping simulation...")
            engine.stop_simulation()

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
    mujoco = MujocoConnector("assets/MuJoCo scene.xml")
    print("Loaded.")

    engine = mujoco

    print("Loading Virtual Reality...")
    with MujocoXRVisualizer(mujoco) as visualizer:
        print("OpenXR visualizer created.")
    # if True:

        #visualizer = MujocoSimpleVisualizer(mujoco)

        print("Connecting to WEART...")
        with WeartConnector() as weart:
            print("Connected. Calibrating...")

            weart.calibrate()
            print("Calibrated.\n")

            simulation(engine, weart, visualizer, None)