from interfaces import GUI, Engine
from pynput import keyboard
from colorama import just_fix_windows_console, Fore, Style

just_fix_windows_console()

class TUI(GUI):
    RESET_KEYS = [keyboard.KeyCode.from_char('R'), keyboard.KeyCode.from_char('r')]
    
    def _key_press(self, key):
        if key == keyboard.Key.esc:
            self._esc_pressed = True
        elif key in TUI.RESET_KEYS:
            print(Fore.YELLOW, "Resetting simulation.", Fore.RESET)
            self._engine.reset_simulation()
        elif key == keyboard.Key.down:
            self._engine.move_finger(1)
    
    def start_gui(self, engine: Engine):
        self._engine = engine

        self._esc_pressed = False
        self._kb_listener = keyboard.Listener(on_press=self._key_press)
        self._kb_listener.start()

        print(Fore.WHITE)
        print(f"Press on {Style.BRIGHT}ESC{Style.NORMAL} to stop the simulation.")
        print(f"Press on {Style.BRIGHT}R{Style.NORMAL} to reset the simulation.")
        print(Style.RESET_ALL)

    def should_exit(self) -> bool:
        return self._esc_pressed
    
    def stop_gui(self):
        self._kb_listener.stop()