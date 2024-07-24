from interfaces import *
from pynput import keyboard
from colorama import just_fix_windows_console, Fore, Style
import numpy as np

just_fix_windows_console()

class TUI(GUI):
    RESET_KEYS = [keyboard.KeyCode.from_char('R'), keyboard.KeyCode.from_char('r')]
    MOVE_KEYS = {
        keyboard.Key.down: [-1, 0, 0],
        keyboard.Key.up: [1, 0, 0],
        keyboard.Key.left: [0, 0, -1],
        keyboard.Key.right: [0, 0, 1]
    }
    MOVE_SCALE = 0.5
    
    def _key_press(self, key):
        if key == keyboard.Key.esc:
            self._esc_pressed = True

        elif key in TUI.RESET_KEYS:
            print(Fore.YELLOW, "Resetting simulation.", Fore.RESET)
            self._engine.reset_simulation()

        elif key in TUI.MOVE_KEYS:
            self._offset_origin += TUI.MOVE_KEYS[key]
            self._visualizer.offset_origin(self._offset_origin * TUI.MOVE_SCALE)
    
    def start_gui(self, engine: Engine, visualizer: Visualizer):
        self._engine = engine
        self._visualizer = visualizer

        self._esc_pressed = False
        self._kb_listener = keyboard.Listener(on_press=self._key_press)
        self._kb_listener.start()

        self._offset_origin = np.zeros(3)

        print(Fore.WHITE)
        print(f"Press on {Style.BRIGHT}ESC{Style.NORMAL} to stop the simulation.")
        print(f"Press on {Style.BRIGHT}R{Style.NORMAL} to reset the simulation.")
        print(f"Press on {Style.BRIGHT}←→↑↓ arrows{Style.NORMAL} to move in the scene.")
        print(Style.RESET_ALL)

    def should_exit(self) -> bool:
        return self._esc_pressed
    
    def stop_gui(self):
        self._kb_listener.stop()