class Engine:
    def move_finger(self, angle : float):
        """Move the virtual finger with an angle

        Args:
            angle (float): new angle of the finger, in radians
        """
        pass
    def move_hand(self, hand_id: int, position: list[int], rotation: list[int]):
        pass
    def get_contact_force(self) -> float:
        """Get the force the finger should feel in contact with the environment

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
    def stop_simulation(self):
        pass

class Visualizer:
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

class HandPoseProvider:
    def get_hand_pose(self, hand_id: int) -> tuple[list[float], list[float]]:
        """Retrieves the hand pose.

        Args:
            hand_id (int): 0 = left hand, 1 = right hand

        Returns:
            tuple[list[float], list[float]]: the position (cartesian)
                and rotation (quaternion) of the hand
        """        
        pass

__all__ = ['Engine', 'Visualizer', 'HandPoseProvider']