#import "@preview/bubble:0.1.0": *
#import "@local/codelst:2.0.2": sourcecode, sourcefile


#show: bubble.with(
  title: "MuJoCoXR",
  subtitle: "Linking MuJoCo and a VR environment",
  author: "Youenn Le Jeune",
  affiliation: "INSA Rennes / DIAG Sapienza",
  date: datetime.today().display(),
  year: none,
  class: none,
  logo: {
    image("logo-insa.png")
    image("logo-sapienza.png")
  }
) 
#set text(lang: "en")
// #set par(first-line-indent: 1em)
#set heading(numbering: (..nums) => {
  let num-format
  if nums.pos().len() == 1{
    num-format = "I."
  } else {
    num-format = "I - 1.1."
  }
  numbering(num-format, ..nums)
})

= Introduction and State of the Art

The MuJoCo physics engine is a great tool that contains many useful features that are nowhere to be found in other engines. Such features are, for instance, realistic soft body simulation. Making interactive simulations is possible through the provided API, in C or in Python. It is easy to integrate motion-capture devices and to get feedback from sensors.

However, the rendering pipeline can be difficult to comprehend. There are multiple pre-made visualization solutions: an interactive one for your MJCF files via the "Simulate" #footnote[https://mujoco.readthedocs.io/en/stable/programming/samples.html#sasimulate] code sample, a passive one that you can use in your Python application to show your interactive simulation in real-time #footnote[https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer], a full Renderer Python class #footnote[https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/renderer.py] that can be really useful for notebooks and, if nothing else is suitable, API functions to render directly in an OpenGL context #footnote[https://mujoco.readthedocs.io/en/stable/programming/visualization.html].

As we can see, there is no built-in way to visualize a MuJoCo simulation in a Virtual Reality environment. At the time of writing, some work is being done to add VR support to the #box["Simulate"] application #footnote[https://github.com/google-deepmind/mujoco/pull/1452]. Nevertheless, it is still not the best choice if you want to fully immerse people in your simulation #footnote()[The "Simulate" application contains a lot of UI elements to control the simulation and visualization, which breaks immersion.]. One possible solution is to use the Unity Plug-in for MuJoCo #footnote[https://mujoco.readthedocs.io/en/stable/unity.html] and configure the Unity application to display in VR (_not tested_), but again that is not ideal: we do not necessarily need a whole game engine to run a simple simulation. Thus, we will program our own solution to display a MuJoCo simulation in a VR headset.

Regarding the VR part, there are multiple solutions to render to a headset:
- Directly use the vendor-specific API (Meta Quest, HTC Vive...). This require code to be remade for each new device we want to support.
- Use OpenVR, which contains support for VR headsets from multiple vendors but it is tied to Steam. #footnote[https://github.com/ValveSoftware/openvr]
- Use OpenXR, which is a free standard not tied to any VR company and implemented by all major VR headsets on the market. #footnote[https://www.khronos.org/openxr/]

We have chosen to use OpenXR because it seems to be the standard that will be mainly used in the future. There are Python bindings available #footnote[https://github.com/cmbruns/pyopenxr]. It supports multiple graphics API including OpenGL that we will use because it is what MuJoCo can render to.

To sum up, our solution will render MuJoCo on a VR headset using OpenGL via the OpenXR standard, all of that in Python.

#pagebreak()

= Theory
== Graphics 101
To render something on a screen or a VR headset, computers use graphic cards (GPUs). Those graphic cards receive orders through Graphic APIs such as OpenGL, Direct3D or more recently, Vulkan. Each GPU supports different versions of those APIs, and old GPUs do not even support some APIs.

A Graphic API consists of a large set of instructions related to graphics: draw a line from here to there, clear the screen with this color, draw this texture. OpenGL instructions are all prefixed with `gl` and make heavy use of constants. For instance, a possible instruction is `glClear(GL_COLOR_BUFFER_BIT)`.

Graphic APIs instructions are not only used to _draw_: with the support of _framebuffers_, it is possible for instance to draw on some in-memory texture and then read it to draw on top of another buffer with a different scale. Framebuffers are a "collection" of renderbuffers and textures and can have multiple _attachements_: colors, depth and stencil.

OpenGL is built on the principle of _extensions_.

Graphic APIs work with a _context_: to use their functions, a context must be bound to the thread. It contains references to all GL objects created within it. A context is usually tied to a window. To create those contexts, we usually use dedicated libraries such as GLFW which allows to create a window and attach the associated context to the calling thread.

In Python, there exist a binding for OpenGL: `pyopengl` #footnote[https://pyopengl.sourceforge.net/].

== OpenXR
OpenXR is a standard implemented by pretty much all VR devices. It provides methods for most of the features: displaying images, getting head position in the room, getting controller positions, rendering haptic feedback, enable passthrough for compatible headsets, and so on. For vendor-specific features, extensions are present in OpenXR to use them.

To render images to the eyes, OpenXR uses the concept of _swapchains_. A swapchain is a collection of framebuffers that display sequentially to a screen. It allows to draw on a "back" framebuffer while another "front" one is being displayed on the screen. #footnote[https://raphlinus.github.io/ui/graphics/gpu/2021/10/22/swapchain-frame-pacing.html]\
In this project, we will make use of only one "stereo" swapchain, which will contain one image for both eyes at the same time. It is also possible to have multiple swapchains, e.g. one per eye.

In order to create an OpenXR-compatible application, a precise suite of operations must be followed (see @XR-app-lifecycle for details):

#figure(image("xr-instance-lifecycle.jpg", width: 110%), caption: [Lifecycle of an OpenXR application #footnote[Extract of the #link("https://www.khronos.org/files/openxr-10-reference-guide.pdf")[OpenXR Reference Guide]] <xr-ref-guide>\
#text(size: 0.9em)[_Not all functions are necessarily used._]], placement: auto) <XR-app-lifecycle>

+ Available extensions are fetched to see if the ones needed are present (for instance, the extension that tells OpenXR to use the OpenGL graphics API, the debug utils extension...)
+ An "instance" is created with the application informations and extensions list. Future methods will use this instance.
+ System information is fetched. At this point, we can use various methods to get configuration information about the view system (for instance, the "screen" size).
+ At this point, a graphic context must be created, although there is no need to have a window (unless we want to mirror what the user will see).
+ OpenXR is told which graphics API to use.
+ A "session" is created within the instance, with binding to the graphic context. Future methods will use this session.
+ The swapchain is created with specific color format, size, samples count ...
+ A reference space is created to be used in head and controllers tracking.
+ A projection layer is created for the swapchain, containing the size and offset of the rectangles associated with each eye.
+ If needed, actions are created to interact with controllers.

At this point, the session is ready. We can enter the main loop:

#figure(image("xr-session-lifecycle.png", width: 80%), caption: [Lifecycle of an OpenXR session @xr-ref-guide], placement: auto) <XR-session-lifecycle>

+ Poll events from OpenXR. Update session according to the new state (see @XR-session-lifecycle for details).
+ If the session is in state `READY`, `SYNCHRONIZED`, `FOCUSED` or `VISIBLE`:
  + Wait for the next frame and when ready, begin it.
  + Locate the views to get the eyes positions and update the projection accordingly.
  + Acquire the swapchain image and render to it.
  + Release the image and end the frame.

== MuJoCo
MuJoCo has a whole #link("https://mujoco.readthedocs.io/en/3.2.0/programming/visualization.html")[documentation chapter] dedicated on visualization and rendering that is worth reading. It also contains tips for rendering to a VR headset.
Other than what is explained in this chapter, there is not much knowledge of MuJoCo required to succeed in this project.

#pagebreak()
= Implementation
Most of the OpenXR / OpenGL related code has been inspired by the #link("https://github.com/cmbruns/pyopenxr_examples/blob/main/xr_examples/gl_example.py")[gl_example] pyopenxr example.

A Python file containing the whole source code is available in a #link("https://gist.github.com/SkytAsul/b1a48a31c4f86b65d72bc8edcb122d3f")[Gist on GitHub]. You can also find a version in the @source_code

#let code = read("../mjxr_tests/mujoco_openxr.py")
#let sourcecode-o = sourcecode
#let sourcecode(..args) = {
  set par(justify: false)
  show raw.where(block: false) : it => it.text
  sourcecode-o(..args)
}
#let show-code(..args) = {
  set par(justify: false)
  show raw.where(block: false) : it => it.text
  sourcefile(code, lang: "python", ..args)
}

We will now see step by step how everything work.

== OpenXR initialization
_The related method is `_init_xr`._

The first part of the method is about creating the OpenXR instance with the required extension. A lot of lines (from line 44 to line 63) are dedicated to make debugging work; this is not interesting. Without it, we can re-write most of the method using one single method call:
#sourcecode(numbers-start: 34)[```py
self._xr_instance = xr.create_instance(xr.InstanceCreateInfo(
    enabled_extension_names: [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
    application_info=xr.ApplicationInfo(
        application_name=APP_NAME,
        engine_name="pyopenxr",
        engine_version=xr.PYOPENXR_CURRENT_API_VERSION,
        api_version=xr.Version(1, 0, xr.XR_VERSION_PATCH)
    )
))
```]

`xr.KHR_OPENGL_ENABLE_EXTENSION_NAME` is a constant describing the name of the `XR_KHR_opengl_enable` extension. #footnote[https://registry.khronos.org/OpenXR/specs/1.1-khr/html/xrspec.html#XR_KHR_opengl_enable]
`APP_NAME` is a constant holding the name of our application that should be displayed to the user. It is defined at the beginning of the file.
There is nothing really fancy here.

#show-code(showrange: (71, 80))
Here we do a bunch of checks to ensure everything is fine and avoid weird errors when rendering afterwards. The `enumerate_view_configuration_views` call at line 73 allows to get the image width and height, that we store. We also store an additional `_width_render` field that is simply the double of the normal width: it is the total width of our stereo render target.

The last part of this method is an ugly mixture of Python and C code to tell OpenXR to use OpenGL. We check that it contains no exception, and then we exit the method.

== OpenGL context and window
_The related method is `_init_window`._

#show-code(showrange: (104, 115))

We use GLFW to create a window. This window is not neccessarily visible (see line 109) but even without it, an OpenGL context will be created.\
The `glfw.window_hint` calls are to set the parameters of the window/context to be created. Notably, we disable double-buffering (a swapchain of two images) because we do not really care about the window rendering quality and it can save us the time of image swapping.\
The window is created with half the size of one eye, so it can fit on the screen (because most VR headset have huge resolutions).\
On line 114, we tell glfw to make the OpenGL context current in the current thread.

It is worth noting that we do not give any information on the OpenGL version and profile we want. This is because MuJoCo specifically requires the _Compatibility profile_ (see #link("https://mujoco.readthedocs.io/en/3.2.0/programming/index.html#using-opengl")[this MuJoCo page] and @ctx_obj).

== OpenXR configuration <prepare_xr>
_The related method is `_prepare_xr`._

The first part of the method has nothing really worth noting: it first creates the `graphics_binding` object that must be passed to the OpenXR session and then it creates the actual session using some C/Python code.

#show-code(showrange: (146, 156))

At line 145 we create the swapchain for our stereo image, hence its width being `_width_render` (which is twice one eye's image width). The format has been selected arbitrarily, the best practice would be to enumerate the available formats and select the best one from here. The usage flags #footnote[https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrSwapchainUsageFlagBits.html] contains:
- `TRANSFER_DST` because the swapchain image will be the destination of a pixel transfer operation (seen later)
- `COLOR_ATTACHMENT` because the swapchain image will have colored pixels on it (in most OpenXR applications this is the case)
- `SAMPLED` because the image can be multisampled

The line 155 is there to retrieve the list of images contained in the swapchain: we do it here once instead of doing it for each frame.

#show-code(showrange: (158, 172))

Here we create the projection layer for the swapchain. It is the object that instructs the runtime where to put the rendered image in the virtual user space. It is done in 3 parts:
+ The reference space for the projection is created with the default options (the `STAGE` space type #footnote[https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html] and default orientation and position).
+ One view per eye is created. Both views are attached to the same swapchain (the one we created earlier) but the `image_rect` of the right eye (which defines which part of the swapchain image is displayed) is offset to the right.
+ Finally, the whole projection layer is created.

Finally at line 171, we create an empty OpenGL framebuffer for the swapchain. We will use it later.

== MuJoCo preparation
_The related method is `_prepare_mujoco`._

#show-code(showrange: (178, 194))

The lines 177 and 178 are basic MuJoCo initialization. This can be done somewhere else in the code, even far sooner.

In this method, we mainly initialize the options of the MuJoCo scene and visualization objects so it can create its render context accordingly. This is done at line 190: when initializing the `MjrContext` object, it internally creates the offscreen framebuffer with the parameters we set at lines 184 to 186.

At this point, everything is ready to start the main loop.

== Frame loop - first part
_The related methods are `loop`, `frame`, `_poll_xr_events` and `_wait_xr_frame`._

The main loop structure looks like this:
#sourcecode(numbering: none)[```py
loop:
    poll_events
    if should_quit:
        stop
    
    if try_start_frame:
        make_a_frame
```]

The `poll_events` part is made of this:
#show-code(showrange: (362, 365))
The `poll_events` method of glfw allows to know if the user wants to close the application on the desktop part (for instance, by closing the mirror window). We update the `_should_quit` field accordingly at line 366.\
The `_poll_xr_events` method is fetching all events from the OpenXR instance and, if the event is a `SESSION_STATE_CHANGED` event, it does the following:
#show-code(showrange: (237, 246))
If this is not clear to you, see the Session lifecycle at @XR-session-lifecycle.

If everything is fine and the visualization should not quit, we wait for the next XR frame:
#show-code(showrange: (210, 218))
If the session is in the right state to render a frame, it waits for the frame to be ready (so we do not render faster than the device refresh rate) and _then_ it returns `True`. We do _not_ start the frame now: it's better to do it the moment right before we will actually render something.

== Frame loop - second part
_The related methods are `frame`, `_update_mujoco`, `_update_views`, `render`, `_end_xr_frame`._

This only happens if the session is in the state to render a frame. This is how it goes:
#show-code(showrange: (371, 377))

The `_update_mujoco` method is really simple:
#show-code(showrange: (200, 201))
The line 199 could be done externally, it is not tied to the visualization: it only steps the physics. On the contrary, the call to `mjv_updateScene` at line 200 fetches geometries from the simulation data and stores it in the scene.

The `_update_views` method is the one that takes care of the head tracking. It goes in 3 parts: first, it fetches the `view_states` which contains, for each eye, its position, orientation and field of view. Then, it updates the projection layer accordingly and the two cameras in the MuJoCo scene to follow the eyes. Finally, it tells MuJoCo that all coordinates should be transformed in a certain way (otherwise, the world is tilted to the right).

The `_render` function is important and complex:
#show-code(showrange: (287, 299))
This first part prepares the framebuffer we created at the end of @prepare_xr by attaching the current swapchain image.\
`glBindFramebuffer(GL_FRAMEBUFFER, fbo)` sets the framebuffer object as the one which will receive the read and draw operations.\
`glFramebufferTexture2D` attaches the image as the first color attachement of the framebuffer object.

#show-code(showrange: (302, 303))
The _real_ rendering is done in the line 304. Afterwards, all is left is to copy the final image from MuJoCo's offscreen framebuffer to our own framebuffer, which has the swapchain image attached.

#show-code(showrange: (305, 315))
The first two instructions are to set which framebuffer will be read from and which one will be drawn on.\
`glBlitFramebuffer` is an instruction to "copy" the pixels (the color ones in our case) from the read framebuffer to the draw one. Both framebuffers color attachements have the same size, so we put the same rectangle twice.

The rest of the method is made to downsample the rendered image and then copy it to our mirror window (if needed).

#pagebreak()
= Enhancements
== Real-time simulation
For now, the code does 1 simulation step per render frame. However, due to synchronization made by OpenXR, one frame cannot be _shorter_ than what it is supposed to be, so the framerate does not exceed the refresh rate of the device (for instance, 80Hz for the Oculus Rift S). This means that the simulation will update 80 times per second, no more. If the timestep set in the MuJoCo is not set to 1/80 of a second, the simulation will not be in "real-time".

To fix that, there are two options:
- The easy one is to change the timestep of your MuJoCo model to match the refresh rate. For the Oculus Rift S, you would set the timestep to $1 slash 80 = 0.0125 s$. This however is not ideal because some simulations will not be stable at such a large timestep.
- The harder one is to change the code to do, for each frame, the amount of simulation steps needed to advance the same amount of time the frame should durate. For a frame duration of $Delta t_"frame" = 1 slash f$ and a timestep of $Delta t_"sim"$, you would advance for $ n_"steps" = floor((Delta t_"frame") / (Delta t_"sim")) $
  You can get the frame duration using `_xr_frame_state.predicted_display_period` (in nanoseconds).

== Performance
For now, everything is single-threaded. If the simulation time is long, it can reduce drastically the framerate and lead to uncomfortable VR experience. It is however possible to split our main loop in two different threads: a "simulation" thread and a "render" thread (see @threading-slide).
#figure(box(stroke: black, image("threading-slide.svg")), caption: [How to multithread an OpenXR application #footnote[Extract of the #link("https://www.khronos.org/assets/uploads/developers/library/2018-gdc-webgl-and-gltf/OpenXR-GDC_Mar18.pdf")[OpenXR presentation at GDC 2018]]]) <threading-slide>
As you can see, the two threads will not be entirely parallel but some of their work will definitely be.

To implement this, we first need to define which work will be given to which thread:
- *Simulation thread:* event polling, frame waiting and simulation stepping.
- *Render thread:* view update, frame begin/end, rendering.
*Beware:* until now, we have grouped the simulation stepping and rendering preparation in one single step (`_update_mujoco`). This should now be split: `mj_step` will be kept in the simulation thread, while `mjv_updateScene` will be put in the rendering thread. This is because `mj_step` can be called while the simulation is being rendered, whereas `mjv_updateScene` cannot (because it contains the geometry data to be rendered).

Now we know where to put which work, the only thing left to do is to split the original `frame` function in two different threads, and use a synchronization structure to signal to the render thread to do its work when simulation is over (a `threading.Semaphore` is suited for this).

A race condition might happen when you remove the headset in this multithreaded setup: `xr.end_session` might be called by the simulation thread while a frame is still being rendered by the other thread. To avoid this, use another synchronization structure (a `threading.Condition` is fine).

== Hand tracking
Hand tracking is not included in the demo file, because it is tied in how the hand is represented in the MJCF. However, here are the basic steps to implement it:
- In the MuJoCo model, add a mocap body that will receive the hand position.
- For OpenXR, we have to create an _action_ that will receive data from the controller. To do that, in the Python program after the `_prepare_xr` step, add another step which follow the same steps as in #link("https://github.com/cmbruns/pyopenxr_examples/blob/9c678b89cf5821730b88a7f3f8145572d536e119/xr_examples/track_controller.py#L20-L89")[this example].
- To get the position and orientation in each frame, use thie code:
#sourcecode[```py
xr.sync_actions(self._xr_session, xr.ActionsSyncInfo(active_action_sets = ctypes.pointer(xr.ActiveActionSet(
    action_set=self._action_set,
    subaction_path=xr.NULL_PATH # wildcard to get all actions
))))
space_location = xr.locate_space(
    space=self._action_space,
    base_space=self._xr_projection_layer.space,
    time=self._xr_frame_state.predicted_display_time
)
if (space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT
    and space_location.location_flags & xr.SPACE_LOCATION_ORIENTATION_VALID_BIT):
    
    hand_pos = numpy.zeros(3)
    hand_rot = numpy.zeros(4)
    orientation = [space_location.pose.orientation[3], *space_location.pose.orientation[3]]
    mujoco.mjv_room2model(hand_pos, hand_rot, list(space_location.pose.position), orientation, self._mj_scene)
    return hand_pos, hand_rot
```]
The important part here is to remember to change the quaternion format to the MuJoCo one $(w, x, y, z)$ and to call `mjv_room2model` to automatically apply the transformations defined in `_update_views` to the pose.

== Note on the `ContextObject` provided by _pyopenxr_ <ctx_obj>
The _pyopenxr_ bindings provide a pre-made class to handle most of the instance, session and swapchain work and let us focus on the interesting parts. This class is named `ContextObject`. However, it is not suitable for use in our case for two reasons:
- `ContextObject` creates one swapchain per eye, whereas we want one big swapchain containing both eyes (because this is how we want MuJoCo to render).
  - This issue could have been bypassed by creating a new swapchain ourselves and not using the premade ones. However, we loose a big part of the advantage of using this class in the first hand: simplicity.
- `ContextObject` uses an internal `OpenGLGraphics` class that handles a lot of rendering-related code. However, this class initializes the OpenGL context with the version 4.5 and _Core profile_. As we saw earlier, MuJoCo requires the _Compatibility profile_.
  - This is not bypassable without recompiling all of _pyopenxr_.
For those reasons, we made every OpenXR-related code from the ground up.


#pagebreak()
= Annex - Source Code <source_code>
#set text(size: 9pt)
#show: it => align(center, block(width: 120%, breakable: true, it))
#show-code(frame: none, showrange: (0, 381))