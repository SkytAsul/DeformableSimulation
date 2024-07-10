import xr
import mujoco
import glfw
import platform
import ctypes
import numpy
from OpenGL import GL
from typing import Optional

from interfaces import Visualizer, HandPoseProvider
from mujoco_connector import MujocoConnector

APP_NAME = "Deformable Simulation"
FRUSTUM_NEAR = 0.05
FRUSTUM_FAR = 50

class MujocoXRVisualizer(Visualizer, HandPoseProvider):
    def __init__(self, mj: MujocoConnector, mirror_window = False, debug = False, samples: Optional[int] = None):
        self._mj_connector = mj
        self._mirror_window = mirror_window
        self._debug = debug
        self._samples = samples
        self._should_quit = False
    
    def __enter__(self):
        self._init_xr()
        self._init_window()
        self._prepare_xr_rendering()
        self._prepare_xr_hand_tracking()
        self._prepare_mujoco()
        glfw.make_context_current(None) # To let other threads use the context if needed
        return self

    def _init_xr(self):
        """
        Initializes the OpenXR environment prior to session creation.

        Also fetches informations about the setup, most importantly the render size.
        """
        extensions = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]
        instance_create_info = xr.InstanceCreateInfo(
            application_info=xr.ApplicationInfo(
                application_name=APP_NAME,
                engine_name="pyopenxr",
                engine_version=xr.PYOPENXR_CURRENT_API_VERSION,
                api_version=xr.Version(1, 0, xr.XR_VERSION_PATCH)
            )
        )

        if self._debug:
            def debug_callback_py(severity, _type, data, _user_data):
                print(severity, f"{data.contents.function_name.decode()}: {data.contents.message.decode()}")
                return True

            debug_messenger = xr.DebugUtilsMessengerCreateInfoEXT(
                message_severities=
                    xr.DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                    | xr.DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
                    | xr.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                    | xr.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                message_types=
                    xr.DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                    | xr.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                    | xr.DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
                    | xr.DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT,
                user_callback=xr.PFN_xrDebugUtilsMessengerCallbackEXT(debug_callback_py)
            )
            instance_create_info.next = ctypes.cast(ctypes.pointer(debug_messenger), ctypes.c_void_p)
            extensions.append(xr.EXT_DEBUG_UTILS_EXTENSION_NAME)

        instance_create_info.enabled_extension_names = extensions
        self._xr_instance = xr.create_instance(instance_create_info)

        # The following fetches important informations about the setup
        # (mainly rendering size)
        self._xr_system = xr.get_system(self._xr_instance, xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY))
        assert xr.enumerate_view_configurations(self._xr_instance, self._xr_system)[0] == xr.ViewConfigurationType.PRIMARY_STEREO
        
        views_config = xr.enumerate_view_configuration_views(self._xr_instance, self._xr_system, xr.ViewConfigurationType.PRIMARY_STEREO)
        assert len(views_config) == 2
        assert views_config[0].recommended_image_rect_width == views_config[1].recommended_image_rect_width
        assert views_config[0].recommended_image_rect_height == views_config[1].recommended_image_rect_height
        
        self._width, self._height = views_config[0].recommended_image_rect_width, views_config[0].recommended_image_rect_height
        self._width_render = self._width * 2

        pxrGetOpenGLGraphicsRequirementsKHR = ctypes.cast(
            xr.get_instance_proc_addr(
                self._xr_instance,
                "xrGetOpenGLGraphicsRequirementsKHR",
            ),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR
        )
        graphics_result = pxrGetOpenGLGraphicsRequirementsKHR(
            self._xr_instance,
            self._xr_system,
            ctypes.byref(xr.GraphicsRequirementsOpenGLKHR())
        )
        graphics_result = xr.exception.check_result(xr.Result(graphics_result))
        if graphics_result.is_exception():
            raise graphics_result

    def _init_window(self):
        """
        Initializes the GLFW window (and make it hidden if mirrored mode is disabled).

        Creates the OpenGL context that will be used.
        """
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        glfw.window_hint(glfw.DOUBLEBUFFER, False)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.SAMPLES, 0) # no need for multisampling here, we will resolve ourselves
        if not self._mirror_window:
            glfw.window_hint(glfw.VISIBLE, False)
        self._window_size = [self._width // 2, self._height // 2]
        self._window = glfw.create_window(*self._window_size, APP_NAME, None, None)
        if self._window is None:
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self._window)
        # Attempt to disable vsync on the desktop window or
        # it will interfere with the OpenXR frame loop timing
        glfw.swap_interval(0)
    
    def _prepare_xr_rendering(self):
        """
        Creates the OpenXR session and prepares everything to launch the frames loop.
        """
        if platform.system() == 'Windows':
            from OpenGL import WGL
            graphics_binding = xr.GraphicsBindingOpenGLWin32KHR()
            graphics_binding.h_dc = WGL.wglGetCurrentDC()
            graphics_binding.h_glrc = WGL.wglGetCurrentContext()
        else:
            from OpenGL import GLX
            graphics_binding = xr.GraphicsBindingOpenGLXlibKHR()
            graphics_binding.x_display = GLX.glXGetCurrentDisplay()
            graphics_binding.glx_context = GLX.glXGetCurrentContext()
            graphics_binding.glx_drawable = GLX.glXGetCurrentDrawable()

        self._xr_session = xr.create_session(
            self._xr_instance,
            xr.SessionCreateInfo(
                0,
                self._xr_system,
                next=ctypes.cast(ctypes.pointer(graphics_binding), ctypes.c_void_p)
            )
        )
        self._xr_session_state = xr.SessionState.IDLE

        self._xr_swapchain = xr.create_swapchain(self._xr_session, xr.SwapchainCreateInfo(
            usage_flags=xr.SWAPCHAIN_USAGE_TRANSFER_DST_BIT | xr.SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | xr.SWAPCHAIN_USAGE_SAMPLED_BIT,
            format=GL.GL_RGBA8,
            sample_count=1 if self._samples is None else self._samples,
            array_size=1,
            face_count=1,
            mip_count=1,
            width=self._width_render,
            height=self._height
        ))
        self._xr_swapchain_images = xr.enumerate_swapchain_images(self._xr_swapchain, xr.SwapchainImageOpenGLKHR)

        self._xr_projection_layer = xr.CompositionLayerProjection(
            # Default space params are okay: identity quaternion and zero vector. Let's use them.
            space=xr.create_reference_space(self._xr_session, xr.ReferenceSpaceCreateInfo()),
            views = [xr.CompositionLayerProjectionView(
                sub_image=xr.SwapchainSubImage(
                    swapchain=self._xr_swapchain,
                    image_rect=xr.Rect2Di(
                        extent=xr.Extent2Di(self._width, self._height),
                        offset=None if eye_index == 0 else xr.Offset2Di(x = self._width) # right eye offset
                    )
                )
            ) for eye_index in range(2)]
        )

        self._xr_swapchain_fbo = GL.glGenFramebuffers(1)
    
    def _prepare_xr_hand_tracking(self):
        subaction_path = xr.string_to_path(self._xr_instance, "/user/hand/right")
        self._action_set = xr.create_action_set(
            instance=self._xr_instance,
            create_info=xr.ActionSetCreateInfo(
                action_set_name="default_action_set",
                localized_action_set_name="Default Action Set",
                priority=0,
            ),
        )
        action = xr.create_action(self._action_set, xr.ActionCreateInfo(
            action_type=xr.ActionType.POSE_INPUT,
            action_name="hand_pose",
            localized_action_name="Hand Pose",
            subaction_paths=[subaction_path]
        ))
        xr.suggest_interaction_profile_bindings(self._xr_instance, xr.InteractionProfileSuggestedBinding(
            interaction_profile=xr.string_to_path(self._xr_instance, "/interaction_profiles/khr/simple_controller"),
            suggested_bindings=[xr.ActionSuggestedBinding(
                action=action,
                binding=xr.string_to_path(self._xr_instance, "/user/hand/right/input/grip/pose")
            )]
        ))
        self._action_space = xr.create_action_space(self._xr_session, xr.ActionSpaceCreateInfo(
            action=action,
            subaction_path=subaction_path
        ))
        xr.attach_session_action_sets(self._xr_session, xr.SessionActionSetsAttachInfo(action_sets=[self._action_set]))

    def _prepare_mujoco(self):
        """
        Prepares the MuJoCo environment.
        """
        self._mj_model = self._mj_connector.model
        self._mj_data = self._mj_connector.data
        self._mj_scene = mujoco.MjvScene(self._mj_model, 1000)
        self._mj_scene.stereo = mujoco.mjtStereo.mjSTEREO_SIDEBYSIDE

        # We want the visualization properties set BEFORE creation of the context,
        # otherwise we would have to call mjr_resizeOffscreen.
        self._mj_model.vis.global_.offwidth = self._width_render
        self._mj_model.vis.global_.offheight = self._height
        self._mj_model.vis.quality.offsamples = 0 if self._samples is None else self._samples

        self._mj_context = mujoco.MjrContext(self._mj_model, mujoco.mjtFontScale.mjFONTSCALE_100)
        self._mj_camera = mujoco._structs.MjvCamera()
        self._mj_option = mujoco.MjvOption()
        # We do NOT want to call mjv_defaultFreeCamera

        mujoco.mjv_defaultOption(self._mj_option)
    
    def _update_mujoco(self):
        """
        Updates MuJoCo for one frame.
        """
        # The simulation is being stepped outside
        mujoco.mjv_updateScene(self._mj_model, self._mj_data, self._mj_option, None, self._mj_camera, mujoco.mjtCatBit.mjCAT_ALL, self._mj_scene)

    def _start_xr_frame(self):
        """
        Starts a frame in the OpenXR environment.

        Returns:
            bool: whether or not we should update the scene and maybe render it.
        """
        if self._xr_session_state in [
            xr.SessionState.READY,
            xr.SessionState.FOCUSED,
            xr.SessionState.SYNCHRONIZED,
            xr.SessionState.VISIBLE,
        ]:
            if self._xr_session_state == xr.SessionState.FOCUSED:
                self._fetch_actions()
            
            self._xr_frame_state = xr.wait_frame(self._xr_session, xr.FrameWaitInfo())
            xr.begin_frame(self._xr_session, None)
            return True
        return False

    def _end_xr_frame(self):
        xr.end_frame(self._xr_session, xr.FrameEndInfo(
            self._xr_frame_state.predicted_display_time,
            xr.EnvironmentBlendMode.OPAQUE,
            layers=[ctypes.byref(self._xr_projection_layer)] if self._xr_frame_state.should_render else []
        ))

    def _poll_xr_events(self):
        while True:
            try:
                event_buffer = xr.poll_event(self._xr_instance)
                event_type = xr.StructureType(event_buffer.type)
                if event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                    event = ctypes.cast(
                        ctypes.byref(event_buffer),
                        ctypes.POINTER(xr.EventDataSessionStateChanged)).contents
                    self._xr_session_state = xr.SessionState(event.state)
                    match self._xr_session_state:
                        case xr.SessionState.READY:
                            if not self._should_quit:
                                xr.begin_session(self._xr_session, xr.SessionBeginInfo(xr.ViewConfigurationType.PRIMARY_STEREO))
                        case xr.SessionState.STOPPING:
                            xr.end_session(self._xr_session)
                            self._should_quit = True
                        case xr.SessionState.EXITING | xr.SessionState.LOSS_PENDING:
                            self._should_quit = True
            except xr.EventUnavailable:
                break # We got all events

    def _update_views(self):
        _, view_states = xr.locate_views(self._xr_session, xr.ViewLocateInfo(
            xr.ViewConfigurationType.PRIMARY_STEREO,
            self._xr_frame_state.predicted_display_time,
            self._xr_projection_layer.space,
        ))
        for eye_index, view_state in enumerate(view_states):
            self._xr_projection_layer.views[eye_index].fov = view_state.fov
            self._xr_projection_layer.views[eye_index].pose = view_state.pose

            cam = self._mj_scene.camera[eye_index]
            cam.pos = list(view_state.pose.position)
            cam.frustum_near = FRUSTUM_NEAR
            cam.frustum_far = FRUSTUM_FAR
            cam.frustum_bottom = numpy.tan(view_state.fov.angle_down) * FRUSTUM_NEAR
            cam.frustum_top = numpy.tan(view_state.fov.angle_up) * FRUSTUM_NEAR
            cam.frustum_center = 0.5 * (numpy.tan(view_state.fov.angle_left) + numpy.tan(view_state.fov.angle_right)) * FRUSTUM_NEAR
            # no need to set left/right as it will be computed using center

            rot_quat = list(view_state.pose.orientation)
            # Guess what? OpenXR quaternions are in form (x, y, z, w)
            # while MuJoCo quaternions are in form (w, x, y, z)...
            rot_quat = quat_xr2mj(rot_quat)

            forward, up = numpy.zeros(3), numpy.zeros(3)
            mujoco.mju_rotVecQuat(forward, [0, 0, -1], rot_quat)
            mujoco.mju_rotVecQuat(up, [0, 1, 0], rot_quat)
            cam.forward, cam.up = forward.tolist(), up.tolist()
        
        self._mj_scene.enabletransform = True
        self._mj_scene.rotate[0] = numpy.cos(0.25 * numpy.pi)
        self._mj_scene.rotate[1] = numpy.sin(-0.25 * numpy.pi)

    def _fetch_actions(self):
        xr.sync_actions(self._xr_session, xr.ActionsSyncInfo(active_action_sets = ctypes.pointer(xr.ActiveActionSet(
            action_set=self._action_set,
            subaction_path=xr.NULL_PATH # wildcard to get all actions
        ))))

    def _render(self):
        """
        Renders the scene in the swapchain and eventually mirrors it on the window if needed.
        """
        # We first ask to acquire a swapchain image to render onto
        image_index = xr.acquire_swapchain_image(self._xr_swapchain, xr.SwapchainImageAcquireInfo())
        xr.wait_swapchain_image(self._xr_swapchain, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))

        # Once we acquired it, we bind the image to our framebuffer object
        glfw.make_context_current(self._window)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._xr_swapchain_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D if self._samples == None else GL.GL_TEXTURE_2D_MULTISAMPLE,
            self._xr_swapchain_images[image_index].image,
            0
        )
        
        # We ask MuJoCo to render on its own offscreen framebuffer
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._mj_context)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, self._width_render, self._height), self._mj_scene, self._mj_context)

        # We copy what MuJoCo rendered on our framebuffer object
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._mj_context.offFBO)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._xr_swapchain_fbo)
        GL.glBlitFramebuffer(
            0, 0,
            self._width_render, self._height,
            0, 0,
            self._width_render, self._height,
            GL.GL_COLOR_BUFFER_BIT,
            GL.GL_NEAREST
        )

        if self._mirror_window:
            # We copy the data from the MuJoCo buffer to the window one (0 is the default window fbo)
            if self._samples is not None:
                # We first resolve multi-sample if needed
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._mj_context.offFBO_r)
                GL.glBlitFramebuffer(
                    0, 0,
                    self._width_render, self._height,
                    0, 0,
                    self._width_render, self._height,
                    GL.GL_COLOR_BUFFER_BIT,
                    GL.GL_NEAREST
                )
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._mj_context.offFBO_r)

            # We then copy the data to the window
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
            GL.glBlitFramebuffer(
                0, 0,
                self._width, self._height, # one eye only (left)
                0, 0,
                *self._window_size,
                GL.GL_COLOR_BUFFER_BIT,
                0x90BA # EXT_framebuffer_multisample_blit_scaled, SCALED_RESOLVE_FASTEST_EXT
            )
        xr.release_swapchain_image(self._xr_swapchain, xr.SwapchainImageReleaseInfo())
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._window is not None:
            glfw.make_context_current(self._window)
            if self._xr_swapchain_fbo is not None:
                GL.glDeleteFramebuffers(1, [self._xr_swapchain_fbo])
                self._xr_swapchain_fbo = None
            glfw.terminate()
        if self._xr_swapchain is not None:
            xr.destroy_swapchain(self._xr_swapchain)
        if self._xr_session is not None:
            xr.destroy_session(self._xr_session)
        if self._xr_instance is not None:
            # # may break on Linux SteamVR
            # xr.destroy_instance(self._xr_instance)
            pass # does not seem to work
        glfw.terminate()

    def start_visualization(self):
        glfw.make_context_current(self._window)

    def stop_visualization(self):
        glfw.make_context_current(None)

    def start_frame(self):
        glfw.poll_events()
        self._poll_xr_events()
        if glfw.window_should_close(self._window):
            self._should_quit = True

        if self._should_quit:
            return (False, None)
        
        if not self._start_xr_frame():
            return (False, None)
        
        return (True, self._xr_frame_state.predicted_display_period / 1000000000) 

    def render_frame(self):
        self._update_mujoco()
        self._update_views()
        if self._xr_frame_state.should_render:
            self._render()
        self._end_xr_frame()

    def should_exit(self):
        return self._should_quit
    
    def get_hand_pose(self, hand_id: int):
        space_location = xr.locate_space(
            space=self._action_space,
            base_space=self._xr_projection_layer.space,
            time=self._xr_frame_state.predicted_display_time
        )
        if (space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT
            and space_location.location_flags & xr.SPACE_LOCATION_ORIENTATION_VALID_BIT):
            
            hand_pos = numpy.zeros(3)
            hand_rot = numpy.zeros(4)
            mujoco.mjv_room2model(hand_pos, hand_rot, list(space_location.pose.position), quat_xr2mj(space_location.pose.orientation), self._mj_scene)
            return hand_pos, hand_rot
        else:
            return None

@staticmethod
def quat_xr2mj(xr_quat):
    return [xr_quat[3], *xr_quat[0:3]]