"""
Second attempt to make a standalone program which displays
a MuJoCo scene on a VR device using openXR.

This attempt is an adaptation of the gl_example from pyopenxr_examples
but with MuJoCo used to render.

https://github.com/cmbruns/pyopenxr_examples/blob/main/xr_examples/gl_example.py
"""

import ctypes
import logging

import glfw
import platform
from OpenGL import GL
if platform.system() == "Windows":
    from OpenGL import WGL
elif platform.system() == "Linux":
    from OpenGL import GLX

import xr
import mujoco

from math import tan, cos, sin, pi
import numpy as np # present in MuJoCo

ALL_SEVERITIES = (
    xr.DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
    | xr.DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
    | xr.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
    | xr.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
)

ALL_TYPES = (
    xr.DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
    | xr.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
    | xr.DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
    | xr.DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT
)


def py_log_level(severity_flags: int):
    if severity_flags & 0x0001:  # VERBOSE
        return logging.DEBUG
    if severity_flags & 0x0010:  # INFO
        return logging.INFO
    if severity_flags & 0x0100:  # WARNING
        return logging.WARNING
    if severity_flags & 0x1000:  # ERROR
        return logging.ERROR
    return logging.CRITICAL


stringForFormat = {
    GL.GL_COMPRESSED_R11_EAC: "COMPRESSED_R11_EAC",
    GL.GL_COMPRESSED_RED_RGTC1: "COMPRESSED_RED_RGTC1",
    GL.GL_COMPRESSED_RG_RGTC2: "COMPRESSED_RG_RGTC2",
    GL.GL_COMPRESSED_RG11_EAC: "COMPRESSED_RG11_EAC",
    GL.GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT: "COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT",
    GL.GL_COMPRESSED_RGB8_ETC2: "COMPRESSED_RGB8_ETC2",
    GL.GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2: "COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2",
    GL.GL_COMPRESSED_RGBA8_ETC2_EAC: "COMPRESSED_RGBA8_ETC2_EAC",
    GL.GL_COMPRESSED_SIGNED_R11_EAC: "COMPRESSED_SIGNED_R11_EAC",
    GL.GL_COMPRESSED_SIGNED_RG11_EAC: "COMPRESSED_SIGNED_RG11_EAC",
    GL.GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM: "COMPRESSED_SRGB_ALPHA_BPTC_UNORM",
    GL.GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC: "COMPRESSED_SRGB8_ALPHA8_ETC2_EAC",
    GL.GL_COMPRESSED_SRGB8_ETC2: "COMPRESSED_SRGB8_ETC2",
    GL.GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2: "COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2",
    GL.GL_DEPTH_COMPONENT16: "DEPTH_COMPONENT16",
    GL.GL_DEPTH_COMPONENT24: "DEPTH_COMPONENT24",
    GL.GL_DEPTH_COMPONENT32: "DEPTH_COMPONENT32",
    GL.GL_DEPTH_COMPONENT32F: "DEPTH_COMPONENT32F",
    GL.GL_DEPTH24_STENCIL8: "DEPTH24_STENCIL8",
    GL.GL_R11F_G11F_B10F: "R11F_G11F_B10F",
    GL.GL_R16_SNORM: "R16_SNORM",
    GL.GL_R16: "R16",
    GL.GL_R16F: "R16F",
    GL.GL_R16I: "R16I",
    GL.GL_R16UI: "R16UI",
    GL.GL_R32F: "R32F",
    GL.GL_R32I: "R32I",
    GL.GL_R32UI: "R32UI",
    GL.GL_R8_SNORM: "R8_SNORM",
    GL.GL_R8: "R8",
    GL.GL_R8I: "R8I",
    GL.GL_R8UI: "R8UI",
    GL.GL_RG16_SNORM: "RG16_SNORM",
    GL.GL_RG16: "RG16",
    GL.GL_RG16F: "RG16F",
    GL.GL_RG16I: "RG16I",
    GL.GL_RG16UI: "RG16UI",
    GL.GL_RG32F: "RG32F",
    GL.GL_RG32I: "RG32I",
    GL.GL_RG32UI: "RG32UI",
    GL.GL_RG8_SNORM: "RG8_SNORM",
    GL.GL_RG8: "RG8",
    GL.GL_RG8I: "RG8I",
    GL.GL_RG8UI: "RG8UI",
    GL.GL_RGB10_A2: "RGB10_A2",
    GL.GL_RGB8: "RGB8",
    GL.GL_RGB9_E5: "RGB9_E5",
    GL.GL_RGBA16_SNORM: "RGBA16_SNORM",
    GL.GL_RGBA16: "RGBA16",
    GL.GL_RGBA16F: "RGBA16F",
    GL.GL_RGBA16I: "RGBA16I",
    GL.GL_RGBA16UI: "RGBA16UI",
    GL.GL_RGBA2: "RGBA2",
    GL.GL_RGBA32F: "RGBA32F",
    GL.GL_RGBA32I: "RGBA32I",
    GL.GL_RGBA32UI: "RGBA32UI",
    GL.GL_RGBA8_SNORM: "RGBA8_SNORM",
    GL.GL_RGBA8: "RGBA8",
    GL.GL_RGBA8I: "RGBA8I",
    GL.GL_RGBA8UI: "RGBA8UI",
    GL.GL_SRGB8_ALPHA8: "SRGB8_ALPHA8",
    GL.GL_SRGB8: "SRGB8",
    GL.GL_RGB16F: "RGB16F",
    GL.GL_DEPTH32F_STENCIL8: "DEPTH32F_STENCIL8",
    GL.GL_BGR: "BGR (Out of spec)",
    GL.GL_BGRA: "BGRA (Out of spec)",
}


class OpenXrExample(object):
    def __init__(self, log_level=logging.DEBUG):
        logging.basicConfig()
        self.logger = logging.getLogger("gl_example")
        self.logger.setLevel(log_level)
        self.debug_callback = xr.PFN_xrDebugUtilsMessengerCallbackEXT(self.debug_callback_py)
        self.mirror_window = False
        self.instance = None
        self.system_id = None
        self.pxrCreateDebugUtilsMessengerEXT = None
        self.pxrDestroyDebugUtilsMessengerEXT = None
        self.pxrGetOpenGLGraphicsRequirementsKHR = None
        self.graphics_requirements = xr.GraphicsRequirementsOpenGLKHR()
        if platform.system() == 'Windows':
            self.graphics_binding = xr.GraphicsBindingOpenGLWin32KHR()
        elif platform.system() == 'Linux':
            self.graphics_binding = xr.GraphicsBindingOpenGLXlibKHR()
        else:
            raise NotImplementedError('Unsupported platform')
        self.render_target_size = None
        self.window = None
        self.session = None
        self.projection_layer_views = (xr.CompositionLayerProjectionView * 2)(
            *([xr.CompositionLayerProjectionView()] * 2))
        self.projection_layer = xr.CompositionLayerProjection(
            views=self.projection_layer_views)
        self.swapchain_create_info = xr.SwapchainCreateInfo()
        self.swapchain = None
        self.swapchain_images = None
        self.quit = False
        self.session_state = xr.SessionState.IDLE
        self.frame_state = xr.FrameState()
        self.eye_view_states = None
        self.window_size = None
        self.enable_debug = True
        self.linux_steamvr_broken_destroy_instance = False

    def debug_callback_py(
            self,
            severity: xr.DebugUtilsMessageSeverityFlagsEXT,
            _type: xr.DebugUtilsMessageTypeFlagsEXT,
            data: ctypes.POINTER(xr.DebugUtilsMessengerCallbackDataEXT),
            _user_data: ctypes.c_void_p,
    ) -> bool:
        d = data.contents
        # TODO structure properties to return unicode strings
        self.logger.log(py_log_level(severity), f"{d.function_name.decode()}: {d.message.decode()}")
        return True

    def run(self):
        while not self.quit:
            if glfw.window_should_close(self.window):
                self.quit = True
            else:
                self.frame()

    def __enter__(self):
        self.prepare_xr_instance()
        self.prepare_xr_system()
        self.prepare_window()
        self.prepare_xr_session()
        self.prepare_xr_swapchain()
        self.prepare_xr_composition_layers()
        self.prepare_gl_framebuffer()
        self.prepare_mujoco()
        return self

    def prepare_xr_instance(self):
        discovered_extensions = xr.enumerate_instance_extension_properties()
        if xr.EXT_DEBUG_UTILS_EXTENSION_NAME not in discovered_extensions:
            self.enable_debug = False
        requested_extensions = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]
        if self.enable_debug:
            requested_extensions.append(xr.EXT_DEBUG_UTILS_EXTENSION_NAME)
        for extension in requested_extensions:
            assert extension in discovered_extensions
        app_info = xr.ApplicationInfo(
            application_name="gl_example",
            application_version=0,
            engine_name="pyopenxr",
            engine_version=xr.PYOPENXR_CURRENT_API_VERSION,
            api_version=xr.Version(1, 0, xr.XR_VERSION_PATCH),
        )
        ici = xr.InstanceCreateInfo(
            application_info=app_info,
            enabled_extension_names=requested_extensions,
        )
        dumci = xr.DebugUtilsMessengerCreateInfoEXT()
        if self.enable_debug:
            dumci.message_severities = ALL_SEVERITIES
            dumci.message_types = ALL_TYPES
            dumci.user_data = None  # TODO
            dumci.user_callback = self.debug_callback
            ici.next = ctypes.cast(ctypes.pointer(dumci), ctypes.c_void_p)  # TODO: yuck
        self.instance = xr.create_instance(ici)
        # TODO: pythonic wrapper
        self.pxrGetOpenGLGraphicsRequirementsKHR = ctypes.cast(
            xr.get_instance_proc_addr(
                self.instance,
                "xrGetOpenGLGraphicsRequirementsKHR",
            ),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR
        )
        instance_props = xr.get_instance_properties(self.instance)
        if platform.system() == 'Linux' and instance_props.runtime_name == b"SteamVR/OpenXR":
            print("SteamVR/OpenXR on Linux detected, enabling workarounds")
            # Enabling workaround for https://github.com/ValveSoftware/SteamVR-for-Linux/issues/422,
            # and https://github.com/ValveSoftware/SteamVR-for-Linux/issues/479
            # destroy_instance() causes SteamVR to hang and never recover
            self.linux_steamvr_broken_destroy_instance = True

    def prepare_xr_system(self):
        get_info = xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY)
        self.system_id = xr.get_system(self.instance, get_info)  # TODO: not a pointer
        view_configs = xr.enumerate_view_configurations(self.instance, self.system_id)
        assert view_configs[0] == xr.ViewConfigurationType.PRIMARY_STEREO.value  # TODO: equality...
        view_config_views = xr.enumerate_view_configuration_views(
            self.instance, self.system_id, xr.ViewConfigurationType.PRIMARY_STEREO)
        assert len(view_config_views) == 2
        assert view_config_views[0].recommended_image_rect_height == view_config_views[1].recommended_image_rect_height
        self.render_target_size = (
            view_config_views[0].recommended_image_rect_width * 2,
            view_config_views[0].recommended_image_rect_height)
        result = self.pxrGetOpenGLGraphicsRequirementsKHR(
            self.instance, self.system_id, ctypes.byref(self.graphics_requirements))  # TODO: pythonic wrapper
        result = xr.exception.check_result(xr.Result(result))
        if result.is_exception():
            raise result

    def prepare_window(self):
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        if not self.mirror_window:
            glfw.window_hint(glfw.VISIBLE, False)
            glfw.window_hint(glfw.SAMPLES, 0)
        glfw.window_hint(glfw.DOUBLEBUFFER, False)
        self.window_size = [self.render_target_size[0] // 2, self.render_target_size[1] // 2]
        self.window = glfw.create_window(*self.window_size, "gl_example", None, None)
        if self.window is None:
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        # Attempt to disable vsync on the desktop window or
        # it will interfere with the OpenXR frame loop timing
        glfw.swap_interval(0)

    def prepare_xr_session(self):
        if platform.system() == 'Windows':
            self.graphics_binding.h_dc = WGL.wglGetCurrentDC()
            self.graphics_binding.h_glrc = WGL.wglGetCurrentContext()
        else:
            self.graphics_binding.x_display = GLX.glXGetCurrentDisplay()
            self.graphics_binding.glx_context = GLX.glXGetCurrentContext()
            self.graphics_binding.glx_drawable = GLX.glXGetCurrentDrawable()
        pp = ctypes.cast(ctypes.pointer(self.graphics_binding), ctypes.c_void_p)
        sci = xr.SessionCreateInfo(0, self.system_id, next=pp)
        self.session = xr.create_session(self.instance, sci)
        reference_spaces = xr.enumerate_reference_spaces(self.session)
        for rs in reference_spaces:
            self.logger.debug(f"Session supports reference space {xr.ReferenceSpaceType(rs)}")
        # TODO: default constructors for Quaternion, Vector3f, Posef, ReferenceSpaceCreateInfo
        rsci = xr.ReferenceSpaceCreateInfo(
            xr.ReferenceSpaceType.STAGE,
            xr.Posef(xr.Quaternionf(0, 0, 0, 1), xr.Vector3f(0, 0, 0))
        )
        self.projection_layer.space = xr.create_reference_space(self.session, rsci)
        swapchain_formats = xr.enumerate_swapchain_formats(self.session)
        for scf in swapchain_formats:
            self.logger.debug(f"Session supports swapchain format {stringForFormat[scf]}")

    def prepare_xr_swapchain(self):
        self.swapchain_create_info.usage_flags = xr.SWAPCHAIN_USAGE_TRANSFER_DST_BIT | xr.SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | xr.SWAPCHAIN_USAGE_SAMPLED_BIT
        self.swapchain_create_info.format = GL.GL_RGBA8
        self.swapchain_create_info.sample_count = 1
        self.swapchain_create_info.array_size = 1
        self.swapchain_create_info.face_count = 1
        self.swapchain_create_info.mip_count = 1
        self.swapchain_create_info.width = self.render_target_size[0]
        self.swapchain_create_info.height = self.render_target_size[1]
        self.swapchain = xr.create_swapchain(self.session, self.swapchain_create_info)
        self.swapchain_images = xr.enumerate_swapchain_images(self.swapchain, xr.SwapchainImageOpenGLKHR)
        for i, si in enumerate(self.swapchain_images):
            self.logger.debug(f"Swapchain image {i} type = {xr.StructureType(si.type)}")

    def prepare_xr_composition_layers(self):
        self.projection_layer.view_count = 2
        self.projection_layer.views = self.projection_layer_views
        for eye_index in range(2):
            layer_view = self.projection_layer_views[eye_index]
            layer_view.sub_image.swapchain = self.swapchain
            layer_view.sub_image.image_rect.extent = xr.Extent2Di(
                self.render_target_size[0] // 2,
                self.render_target_size[1],
            )   
            # the following causes MuJoCo to render blank to the right eye
            if eye_index == 1:
                layer_view.sub_image.image_rect.offset.x = layer_view.sub_image.image_rect.extent.width

    def prepare_gl_framebuffer(self):
        w, h = self.swapchain_create_info.width, self.swapchain_create_info.height
        glfw.make_context_current(self.window)

        # FBO
        self.sw_fbo = GL.glGenFramebuffers(1)
        # GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.sw_fbo)
        # GL.glViewport(0, 0, w, h)

        # COLOR


        # DEPTH
        """ self.fbo_depth_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.fbo_depth_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT32, w, h, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)

        if self.mirror_window:
            self.fbo_downsample = GL.glGenBuffers(1)
            self.tex_dowmsample = GL.glGenTextures(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_downsample)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_dowmsample)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, *self.window_size, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.tex_dowmsample, 0)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0) """

    def prepare_mujoco(self):
        self.first = True
        self._model = mujoco.MjModel.from_xml_path("assets/balloons.xml")
        self._data = mujoco.MjData(self._model)
        self._scene = mujoco.MjvScene(self._model, 1000)
        self._context = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_100)
        self._camera = mujoco._structs.MjvCamera()
        self._camera.azimuth = 0
        self._camera.distance = 0
        self._camera.elevation = 0
        self._camera.lookat = [0, 0, 0]
        # The following sets default things we do not want
        # mujoco.mjv_defaultFreeCamera(self._model, self._camera)
        self._model.vis.global_.offwidth = self.render_target_size[0]
        self._model.vis.global_.offheight = self.render_target_size[1]
        mujoco.mjr_resizeOffscreen(*self.render_target_size, self._context)

        self._option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self._option)

    def frame(self):
        glfw.poll_events()
        self.poll_xr_events()
        if self.quit:
            return
        if self.start_xr_frame():
            mujoco.mj_step(self._model, self._data)
            mujoco.mjv_updateScene(self._model, self._data, self._option, None, self._camera, mujoco.mjtCatBit.mjCAT_ALL, self._scene)

            self.update_xr_views()
            if self.frame_state.should_render:
                self.render()
            self.end_xr_frame()

    def poll_xr_events(self):
        while True:
            try:
                event_buffer = xr.poll_event(self.instance)
                event_type = xr.StructureType(event_buffer.type)
                if event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                    self.on_session_state_changed(event_buffer)
            except xr.EventUnavailable:
                break

    def on_session_state_changed(self, session_state_changed_event):
        # TODO: it would be nice to avoid this horrible cast...
        event = ctypes.cast(
            ctypes.byref(session_state_changed_event),
            ctypes.POINTER(xr.EventDataSessionStateChanged)).contents
        # TODO: enum property
        self.session_state = xr.SessionState(event.state)
        if self.session_state == xr.SessionState.READY:
            if not self.quit:
                sbi = xr.SessionBeginInfo(xr.ViewConfigurationType.PRIMARY_STEREO)
                xr.begin_session(self.session, sbi)
        elif self.session_state == xr.SessionState.STOPPING:
            xr.end_session(self.session)
            self.session = None
            self.quit = True

    def start_xr_frame(self) -> bool:
        if self.session_state in [
            xr.SessionState.READY,
            xr.SessionState.FOCUSED,
            xr.SessionState.SYNCHRONIZED,
            xr.SessionState.VISIBLE,
        ]:
            frame_wait_info = xr.FrameWaitInfo(None)
            try:
                self.frame_state = xr.wait_frame(self.session, frame_wait_info)
                xr.begin_frame(self.session, None)
                return True
            except xr.ResultException as ex:
                print(ex)
                return False
        return False

    def end_xr_frame(self):
        frame_end_info = xr.FrameEndInfo(
            self.frame_state.predicted_display_time,
            xr.EnvironmentBlendMode.OPAQUE
        )
        if self.frame_state.should_render:
            for eye_index in range(2):
                layer_view = self.projection_layer_views[eye_index]
                eye_view = self.eye_view_states[eye_index]
                layer_view.fov = eye_view.fov
                layer_view.pose = eye_view.pose
            frame_end_info.layers = [ctypes.byref(self.projection_layer), ]
        xr.end_frame(self.session, frame_end_info)

    @staticmethod
    def quat_xr2mj(xr_quat):
        return [xr_quat[3], *xr_quat[0:3]]

    def update_xr_views(self):
        vi = xr.ViewLocateInfo(
            xr.ViewConfigurationType.PRIMARY_STEREO,
            self.frame_state.predicted_display_time,
            self.projection_layer.space,
        )
        vs, self.eye_view_states = xr.locate_views(self.session, vi)
        for eye_index, view_state in enumerate(self.eye_view_states):
            rot_quat = list(view_state.pose.orientation)

            self._scene.stereo = mujoco.mjtStereo.mjSTEREO_SIDEBYSIDE
            cam = self._scene.camera[eye_index]
            cam.pos = list(view_state.pose.position)
            FRUSTUM_NEAR = 0.05
            FRUSTUM_FAR = 50
            cam.frustum_near = FRUSTUM_NEAR
            cam.frustum_far = FRUSTUM_FAR
            cam.frustum_bottom = tan(view_state.fov.angle_down) * FRUSTUM_NEAR
            cam.frustum_top = tan(view_state.fov.angle_up) * FRUSTUM_NEAR
            cam.frustum_center = 0.5 * (tan(view_state.fov.angle_left) + tan(view_state.fov.angle_right)) * FRUSTUM_NEAR
            # no need to set left/right as it will be computed using center

            forward = np.zeros(3)
            up = np.zeros(3)
            mujoco.mju_rotVecQuat(forward, [0, 0, -1], OpenXrExample.quat_xr2mj(rot_quat))
            mujoco.mju_rotVecQuat(up, [0, 1, 0], OpenXrExample.quat_xr2mj(rot_quat))
            cam.forward = forward.tolist()
            cam.up = up.tolist()
            # cam.forward = [0, 0, -1]
            # cam.up = [0, 1 if eye_index == 1 else -1, 0]
        
        self._scene.enabletransform = True
        self._scene.translate[0] = 0;
        self._scene.translate[2] = 0;
        self._scene.rotate[0] = cos(0.25 * pi);
        self._scene.rotate[1] = sin(-0.25 * pi);
        self._scene.rotate[2] = 0;
        self._scene.rotate[3] = 0;
        self._scene.scale = 1;

    def render(self):
        ai = xr.SwapchainImageAcquireInfo(None)
        swapchain_index = xr.acquire_swapchain_image(self.swapchain, ai)
        wi = xr.SwapchainImageWaitInfo(xr.INFINITE_DURATION)
        xr.wait_swapchain_image(self.swapchain, wi)
        glfw.make_context_current(self.window)
        w, h = self.render_target_size

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.sw_fbo)
        sw_image = self.swapchain_images[swapchain_index]

        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST);
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST);
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE);
        # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE);
        # # GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None);

        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, sw_image.image, 0)
        # GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self.fbo_depth_tex, 0)

        # print("INFO")
        # print(hex(GL.glGetFramebufferAttachmentParameteriv(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING)))
        # print(hex(GL.glGetFramebufferAttachmentParameteriv(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE)))
        # print(hex(GL.glGetFramebufferAttachmentParameteriv(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE)))

        viewport = mujoco.MjrRect(0, 0, w, h)

        # print("Skybox:", self._scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX])
        # print("Reflection:", self._scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION])

        # GL.glEnable(GL.GL_SCISSOR_TEST)
        # GL.glScissor(0, 0, w // 2, h)
        # GL.glClearColor(0, 1, 0, 1)
        # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # GL.glScissor(w // 2, 0, w // 2, h)
        # GL.glClearColor(0, 0, 1, 1)
        # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # GL.glDisable(GL.GL_SCISSOR_TEST)

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context)
        mujoco.mjr_render(viewport, self._scene, self._context)

        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._context.offFBO)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.sw_fbo)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glBlitFramebuffer(
            0, 0,
            w, h,
            0, 0,
            w, h,
            GL.GL_COLOR_BUFFER_BIT,
            GL.GL_NEAREST
        )

        if self.mirror_window:
            # fast blit from the fbo to the window surface
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.sw_fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
            GL.glBlitFramebuffer(
                0, 0,
                w, h,
                0, 0,
                *self.window_size,
                GL.GL_COLOR_BUFFER_BIT,
                0x90BA # EXT_framebuffer_multisample_blit_scaled
            )

            # GL.glFramebufferTexture(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, 0, 0)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        ri = xr.SwapchainImageReleaseInfo()
        xr.release_swapchain_image(self.swapchain, ri)
        # If we're mirror make sure to do the potentially blocking command
        # AFTER we've released the swapchain image
        if self.mirror_window:
            glfw.swap_buffers(self.window)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.window is not None:
            glfw.make_context_current(self.window)
            if self.sw_fbo is not None:
                GL.glDeleteFramebuffers(1, [self.sw_fbo])
                self.sw_fbo = None
            # if self.fbo_depth_buffer is not None:
            #     GL.glDeleteRenderbuffers(1, [self.fbo_depth_buffer])
            #     self.fbo_depth_buffer = None
            glfw.terminate()
            self.window = None
        if self.swapchain is not None:
            xr.destroy_swapchain(self.swapchain)
            self.swapchain = None
        if self.session is not None:
            xr.destroy_session(self.session)
            self.session = None
        self.system_id = None
        if self.instance is not None:
            # Workaround for https://github.com/ValveSoftware/SteamVR-for-Linux/issues/422
            # and https://github.com/ValveSoftware/SteamVR-for-Linux/issues/479
            if not self.linux_steamvr_broken_destroy_instance:
                xr.destroy_instance(self.instance)
            self.instance = None
        glfw.terminate()


if __name__ == "__main__":
    with OpenXrExample(logging.DEBUG) as ex:
        ex.run()