# Introduction
The goal of this project is to link a haptic device (the WEART TouchDIVER), a VR headset (the Oculus Rift S) and a physics simulation software (CoppeliaSim with the MuJoCo engine) to simulate real-time touch of a deformable object.

This project is conducted at the DIAG Robotics Laboratory of Sapienza University of Rome, under the supervision of Marilena Vendittelli.

# Dependencies
- [WEART Python SDK](https://github.com/WEARTHaptics/WEART-SDK-Python)
    - installable via `pip install weartsdk-sky`
- CoppeliaSim ZMQ API
    - see [the manual](https://manual.coppeliarobotics.com/en/zmqRemoteApiOverview.htm)
- [pyopenxr](https://github.com/cmbruns/pyopenxr/)
    - `pip install pyopenxr`
- [Python keyboard library](https://pypi.org/project/keyboard/)
    - `pip install keyboard`
- [matplotlib](https://pypi.org/project/matplotlib/) (for real-time performance plots)
    - `pip install matplotlib`

# How to use
1. Open the [scene](<assets/CoppeliaSim scene.ttt>) in CoppeliaSim.
1. Open the WEART Middleware and connect your TouchDIVER.
1. Launch the [`simulator.py`](simulator.py) python file.