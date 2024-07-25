# local code
from interfaces import *
from coppelia import CoppeliaConnector
from mujoco_connector import MujocoConnector, MujocoSimpleVisualizer
from mujoco_xr import MujocoXRVisualizer
from weart import WeartConnector, HAPTIC_FINGERS
from guis import TUI
from benchmarking import Benchmarker, Plotter

# libraries
from contextlib import nullcontext
from threading import Thread
from colorama import just_fix_windows_console, Fore, Style

just_fix_windows_console()

def simulation(engine: Engine,
                weart: WeartConnector | None,
                visualizer: Visualizer,
                hand_provider: HandPoseProvider | None,
                gui: GUI,
                haptic_hands: list[int],
                tracking_hands: list[int]):
    print("Starting simulation.")
    engine.start_simulation()

    perf_bench = Benchmarker(title="Performance profiler")
    frame_bench = Benchmarker(title="Performance profiler")
    force_plot = Plotter(title="Applied force graph")

    def loop():
        print("Starting visualization...")
        visualizer.start_visualization()
        gui.start_gui(engine, visualizer)
        if isinstance(visualizer, MujocoXRVisualizer):
            visualizer.add_perf_counters(perf_bench, frame_bench)

        print(Style.BRIGHT + Fore.GREEN, f"Done!{Style.NORMAL} Everything is up and running.{Fore.RESET}\n")
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

                if hand_provider is not None:
                    for hand in tracking_hands:
                        hand_pose = hand_provider.get_hand_pose(hand)
                        if hand_pose is not None:
                            engine.move_hand(hand, *hand_pose)

                engine.step_simulation(frame_duration)
                perf_bench.mark("Step simulation")

                visualizer.render_frame()
                # perf_bench.mark("Render")

                for hand in haptic_hands:
                    for finger in HAPTIC_FINGERS:
                        force = engine.get_contact_force(hand, finger)
                        # perf_bench.mark("Contact force")
                        force_plot.plot(force, f"{finger} hand {hand}")

                        if weart is not None:
                            weart.apply_force(hand, finger, force)
                            # perf_bench.mark("Apply force to finger")

                force_plot.end_iteration()
                perf_bench.end_iteration()
        except KeyboardInterrupt:
            pass # To exit gracefully. Even though we swallow the error, we still exit the loop.
        finally:
            force_plot.stop()
            perf_bench.stop()
            #perf_bench.export_csv("benchmark.csv", include_time=True)

            print("\nStopping visualization...")
            gui.stop_gui()
            visualizer.stop_visualization()

            print("Stopping simulation...")
            engine.stop_simulation()

            print("\nCiao!\n")

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
    # CHANGEABLE PARAMETERS

    used_engine = "mujoco"
    used_viz = "simple"
    use_weart = True
    used_gui = "tui"

    # scene_path = "assets/MuJoCo scene.xml"
    # scene_path = "assets/balloons.xml"
    scene_path = "assets/MuJoCo phantom.xml"

    # Actually, it seems like everything works even if not both hands are connected (WEART and Oculus)
    tracking_hands = {"left": True, "right": True}
    haptic_hands = {"left": True, "right": True}


    # SCRIPT
    print("Starting script...\n")

    enabled_hands_tracking = [i for i, (side, enabled) in enumerate(tracking_hands.items()) if enabled]
    enabled_hands_haptic = [i for i, (side, enabled) in enumerate(haptic_hands.items()) if enabled]

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
            visualizer_ctx = hand = MujocoXRVisualizer(mujoco, mirror_window=True, samples=8, fps_counter=False)
        case _:
            raise RuntimeError("Invalid visualizer name")

    match used_gui:
        case "tui":
            gui = TUI()
        # TODO: add another GUI, maybe a TK window?
        case _:
            raise RuntimeError("Invalid GUI name")

    with visualizer_ctx as visualizer:
        print("Visualizer created.\n")

        if use_weart:
            print("Connecting to WEART...")
            weart_ctx = WeartConnector(enabled_hands_haptic)
        else:
            weart_ctx = nullcontext()
            print(Fore.RED + Style.BRIGHT, "WARNING:", Style.NORMAL + "You have not enabled WEART.\n", Style.RESET_ALL)

        with weart_ctx as weart:
            # everything is initialized at this point
            
            simulation(engine, weart, visualizer, hand, gui, enabled_hands_tracking, enabled_hands_haptic)