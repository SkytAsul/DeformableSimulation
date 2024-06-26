# Introduction
The goal of this project is to link a haptic device (the WEART TouchDIVER), a VR headset (the Oculus Rift S) and a physics simulation software (CoppeliaSim with the MuJoCo engine) to simulate real-time touch of a deformable object.

This project is conducted at the DIAG Robotics Laboratory of Sapienza University of Rome, under the supervision of Marilena Vendittelli.

# Dependencies
- [WEART Python SDK](https://github.com/WEARTHaptics/WEART-SDK-Python) for the TouchDIVER
    - installable via `pip install weartsdk-sky`
- CoppeliaSim ZMQ API for the simulation
    - see [the manual](https://manual.coppeliarobotics.com/en/zmqRemoteApiOverview.htm)
- [pyopenxr](https://github.com/cmbruns/pyopenxr/) for the VR headset and the controllers
    - `pip install pyopenxr`
- [Python keyboard library](https://pypi.org/project/keyboard/)
    - `pip install keyboard`
- [matplotlib](https://pypi.org/project/matplotlib/) for real-time performance plots
    - `pip install matplotlib`

# How to use
1. Open the [scene](<assets/CoppeliaSim scene.ttt>) in CoppeliaSim.
1. Open the WEART Middleware and connect your TouchDIVER.
1. Launch the [`simulator.py`](simulator.py) python file.

# Platforms
The project is currently written in pure Python code and depends on platform-independent libraries. It is therefore cross-platform.  
However, to use a TouchDIVER, the *WEART Middleware* must be opened and this software is Windows-only. There is fortunately a workaround in the following section.

## Use on non-Windows platforms
If you have a secondary computer with Windows installed, it is possible to run the simulation on a primary (Linux for example) computer:
1. Launch CoppeliaSim on the primary computer
1. Launch WEART Middleware on the Windows computer
1. Link the primary and the secondary computers and allow the primary one to interact with the loopback-bound 13031 TCP port
    - Either by putting them in the same network and using a tool like *socat* to redirect the TCP port
    - Either by using [ZeroTierOne](https://github.com/zerotier/ZeroTierOne) as an easy solution (untested, and you will still need to setup the TCP redirection)
    - Either by setting up tunnels if there is no way to do the above
        - This depends on your network configuration.
        - If both computers are "hidden" from each other, for instance between two NATs, you can use a third publicly accessible machine (for instance a cloud VPS) as a relay. Setup a reverse SSH tunnel between the Windows and the relay which exposes the port 13031 to a public one, and make the Linux client connect to the relay's port.
        - `ssh -N -4 -R <relay-port>:localhost:13031 <user>@<relay address>` (the `-4` switch is necessary on Windows, see [this issue](https://github.com/PowerShell/Win32-OpenSSH/issues/1265))