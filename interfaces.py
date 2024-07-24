"""Contains the classes that must be implemented to have a complete simulation.
"""

class Engine:
    """Represents a link to a physics engine.
    """
    def move_finger(self, angle : float):
        """Move the virtual finger with an angle

        Args:
            angle (float): new angle of the finger, in radians
        """
        pass
    def move_hand(self, hand_id: int, position: list[float], rotation: list[float]):
        """Move the virtual hand in the simulation

        Args:
            hand_id (int): hand_id (int): 0 = left hand, 1 = right hand
            position (list[float]): cartesian position of the hand
            rotation (list[float]): orientation quaternion of the hand
        """
        pass
    def get_contact_force(self, hand_id: int, finger: str) -> float:
        """Get the force the finger should feel in contact with the environment

        Args:
            hand_id (int): 0 = left hand, 1 = right hand
            finger (str): thumb, index, middle, annular or pinky

        Returns:
            float: a value between 0 and 1
        """
        pass
    def start_simulation(self):
        pass
    def step_simulation(self, duration: float | None):
        """Steps the simulation for the amount of time specified.

        Args:
            duration (float | None): amount of seconds to simulate.
            If None, then only step once.
        """        
        pass
    def reset_simulation(self):
        """Resets the simulation to its original state.
        """
        pass
    def stop_simulation(self):
        pass

class Visualizer:
    """Represents a way to visualize the simulation.
    """
    def start_visualization(self):
        pass
    def wait_frame(self) -> tuple[bool, float | None]:
        """Starts a render frame.

        Returns:
            tuple[bool, float | None]: a tuple containing True and the amount of
            seconds this frame should durate (or None if unknown) IF the frame
            should render, a tuple containing False and any other value otherwise.
        """
        return (True, None)
    def render_frame(self):
        pass
    def should_exit(self) -> bool:
        return False
    def stop_visualization(self):
        pass
    def offset_origin(self, position: list[float]):
        """Offsets the origin position of the visualization.

        Args:
            position (list[float]): a list of 3 floats describing the new offset (overrides the previous one)
        """
        pass

class HandPoseProvider:
    """Represents an object that can provide the position of hands in real-time.
    """
    def get_hand_pose(self, hand_id: int) -> tuple[list[float], list[float]]:
        """Retrieves the hand pose.

        Args:
            hand_id (int): 0 = left hand, 1 = right hand

        Returns:
            tuple[list[float], list[float]]: the position (cartesian)
                and rotation (quaternion) of the hand
        """        
        pass

class GUI:
    """Represents a way for the user to interact with the simulation.
    """
    def start_gui(self, engine: Engine, visualizer: Visualizer):
        pass
    def should_exit(self) -> bool:
        return False
    def stop_gui(self):
        pass

__all__ = ['Engine', 'Visualizer', 'HandPoseProvider', 'GUI']