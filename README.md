# Introduction
The goal of this project is to link a haptic device (the WEART TouchDIVER), a VR headset (the Oculus Rift S) and a physics simulation software (CoppeliaSim with the MuJoCo engine) to simulate real-time touch of a deformable object.

This project is conducted at the DIAG Robotics Laboratory of Sapienza University of Rome, under the supervision of Marilena Vendittelli.

# MuJoCoXR
During the project, a way to display MuJoCo in a VR headset has been developed.

You will find the attempts in [mjxr_tests](mjxr_tests). The final, working one is [mujoco_openxr.py](mjxr_tests/mujoco_openxr.py) and is also available on [GitHub Gists](https://gist.github.com/SkytAsul/b1a48a31c4f86b65d72bc8edcb122d3f).

To better understand how everything work, see [the paper I wrote](mujocoxr-paper/MuJoCoXR.pdf).

# Deformable mesh tutorial
I made a tutorial on how to convert a 3D model to a deformable material in MuJoCo. It is available [here](deformable-mesh-tuto/deformable-mesh-tuto.pdf).

# Simulator
## Dependencies
Before installing dependencies, remember to create a Python virtual environment!

- [WEART Python SDK](https://github.com/WEARTHaptics/WEART-SDK-Python)
    - for the TouchDIVER
    - `pip install weartsdk-sky`
- [CoppeliaSim](https://coppeliarobotics.com/) ZMQ API
    - for the simulation
    - `pip install coppeliasim_zmqremoteapi_client`
    - see [the manual](https://manual.coppeliarobotics.com/en/zmqRemoteApiOverview.htm)
- [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html)
    - for the simulation
    - `pip install mujoco`
    - see [the manual](https://mujoco.readthedocs.io/en/stable/python.html)
- [pyopenxr](https://github.com/cmbruns/pyopenxr/)
    - for the VR headset and the controllers
    - `pip install pyopenxr`
- [pynput](https://pypi.org/project/pynput/)
    - to listen to keyboard press
    - `pip install pynput`
- [colorama](https://pypi.org/project/colorama/)
    - for pretty-printing
    - `pip install colorama`
- [matplotlib](https://pypi.org/project/matplotlib/)
    - for real-time performance plots
    - `pip install matplotlib`

You can install all these dependencies at once by executing this command while being in the repository directory:
```sh
$ pip install -r requirements.txt
```

## How to use
### CoppeliaSim simulation
1. Open the [scene](<assets/CoppeliaSim scene.ttt>) in CoppeliaSim.
1. Open the WEART Middleware and connect your TouchDIVER.
1. Change the options in [`simulator.py`](simulator.py) so they match your setup.
1. Launch the [`simulator.py`](simulator.py) python file.

### MuJoCo simulation
1. Open the WEART Middleware and connect your TouchDIVERs.
1. Connect your VR device and launch the runtime program (Meta Quest Link for instance).
1. Change the options in [`simulator.py`](simulator.py) so they match your setup.
1. Launch the [`simulator.py`](simulator.py) python file.

## Platforms
The project is currently written in pure Python code and depends on cross-platform libraries. It is therefore also cross-platform.  
However, to use a TouchDIVER, the *WEART Middleware* must be opened and this software is Windows-only. There is fortunately a workaround in the following section.

### Use on non-Windows platforms
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

Use-case: CoppeliaSim runs *much* faster on Linux.

## MuJoCo simulation scene
You can plug any simulation scene you want. The only requirements are:
- the hands mocap bodies must follow the names `{side}_hand_mocap`, where `{side}` is _left_ or _right_.
- the hands "real" bodies must follow the names `{side}_hand` and their rotations must be expressed with the `euler` parameter.
- the fingertip sensors must be of type `contact` and follow the names `{side}_fingertip_{finger}`, where `{finger}` is _thumb_, _index_ or _middle_.