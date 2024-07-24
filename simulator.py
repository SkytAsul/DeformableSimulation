# local code
from interfaces import *
from coppelia import CoppeliaConnector
from mujoco_connector import MujocoConnector, MujocoSimpleVisualizer
from mujoco_xr import MujocoXRVisualizer
from weart import WeartConnector
from guis import TUI
from benchmarking import Benchmarker, Plotter

# libraries
from contextlib import nullcontext
from threading import Thread
import math

def closure_to_angle(closure):
    # dumb calc: between 0 and 0.4 we are proportional to 0 and 40°
    degree = closure * 100 if closure < 0.4 else 40
    return math.radians(degree)

def simulation(engine: Engine,
                weart: WeartConnector | None,
                visualizer: Visualizer,
                hand: HandPoseProvider | None,
                gui: GUI):
    print("Starting simulation.")
    engine.start_simulation()
    if weart is not None:
        weart.start_listeners()

    perf_bench = Benchmarker(title="Performance profiler")
    frame_bench = Benchmarker(title="Performance profiler")
    force_plot = Plotter(title="Applied force graph")

    def loop():
        print("Starting visualization...")
        visualizer.start_visualization()
        gui.start_gui(engine, visualizer)
        if isinstance(visualizer, MujocoXRVisualizer):
            visualizer.add_perf_counters(perf_bench, frame_bench)

        print("Done! Everything is up and running.\n")
        try:
            while not visualizer.should_exit() and not gui.should_exit():
                frame_bench.new_iteration()
                frame_continue, frame_duration = visualizer.wait_frame()
                frame_bench.mark("Wait frame")
                frame_bench.end_iteration()
                if visualizer.should_exit():
                    break
                if not frame_continue:
                    continue
                perf_bench.new_iteration()
                force_plot.new_iteration()

                if weart is not None:
                    angle = closure_to_angle(weart.get_index_closure())
                    engine.move_finger(angle)
                    perf_bench.mark("Hand movements")

                if hand is not None:
                    hand_pose = hand.get_hand_pose(0)
                    if hand_pose is not None:
                        engine.move_hand(0, *hand_pose)

                engine.step_simulation(frame_duration)
                perf_bench.mark("Step simulation")

                visualizer.render_frame()
                # perf_bench.mark("Render")

                force = engine.get_contact_force()
                # perf_bench.mark("Contact force")
                force_plot.plot(force, "Force")

                if weart is not None:
                    weart.apply_force(force)
                    perf_bench.mark("Apply force to finger")

                force_plot.end_iteration()
                perf_bench.end_iteration()
        except KeyboardInterrupt:
            pass # To exit gracefully. Even though we swallow the error, we still exit the loop.
        finally:
            force_plot.stop()
            perf_bench.stop()
            #perf_bench.export_csv("benchmark.csv", include_time=True)

            print("Stopping visualization...")
            gui.stop_gui()
            visualizer.stop_visualization()

            print("Stopping simulation...")
            engine.stop_simulation()

            print("Ciao!")

    threaded = False
    if threaded:
        t = Thread(target=loop)
        t.start()
        # we must run the loop in another thread because the graph can only be visualized in the main thread...
        #perf_bench.graph_viz(max_points=1000, use_time=True)
        #force_plot.graph_viz(max_points=10000, y_axis="Force")
        t.join()
    else:
        loop()

if __name__ == "__main__":
    used_engine = "mujoco"
    used_viz = "openxr"
    use_weart = False
    used_gui = "tui"
    # scene_path = "assets/MuJoCo scene.xml"
    # scene_path = "assets/balloons.xml"
    scene_path = "assets/MuJoCo phantom.xml"

    print("Starting script...\n")

    engine = visualizer = weart = hand = gui = None

    match used_engine:
        case "mujoco":
            print("Loading MuJoCo...")
            engine = mujoco = MujocoConnector(scene_path)
            print("Loaded.\n")
        case "coppelia":
            print("Connecting to Coppelia...")
            engine = CoppeliaConnector()
            print("Connected.\n")
        case _:
            raise RuntimeError("Invalid engine name")

    match used_viz:
        case "simple":
            visualizer_ctx = nullcontext(MujocoSimpleVisualizer(mujoco))
        case "openxr":
            print("Loading Virtual Reality...")
            visualizer_ctx = hand = MujocoXRVisualizer(mujoco, mirror_window=False, samples=8, fps_counter=False)
        case _:
            raise RuntimeError("Invalid visualizer name")

    match used_gui:
        case "tui":
            gui = TUI()
        case _:
            raise RuntimeError("Invalid GUI name")

    with visualizer_ctx as visualizer:
        print("Visualizer created.\n")

        if use_weart:
            print("Connecting to WEART...")
            weart_ctx = WeartConnector()
        else:
            weart_ctx = nullcontext()

        with weart_ctx as weart:
            
            if use_weart:
                print("Connected. Calibrating, stand still...")

                weart.calibrate()
                print("Calibrated.\n")

            simulation(engine, weart, visualizer, hand, gui)