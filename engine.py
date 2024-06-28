class Engine:
    def move_finger(self, angle : float):
        """Move the virtual finger with an angle

        Args:
            angle (float): new angle of the finger, in radians
        """
        pass
    def get_contact_force(self) -> float:
        """Get the force the finger should feel in contact with the environment

        Returns:
            float: a value between 0 and 1
        """
        pass
    def start_simulation(self):
        pass
    def step_simulation(self):
        pass
    def stop_simulation(self):
        pass