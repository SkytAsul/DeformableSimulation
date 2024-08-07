from dataclasses import dataclass

@dataclass
class Hand:
    id: int
    """ID of the hand. 0 = left, 1 = right"""
    side: str
    """Side, left or right."""
    tracking: bool
    """Whether the space position and rotation of the hand is mapped to the virtual one."""
    haptics: bool
    """Whether we can retrieve finger closures and send haptic feedback to the hand."""
    controller_rotation: float
    """The initial rotation of the hand, in degrees.
    0 means the controller is in the same direction as the hand."""