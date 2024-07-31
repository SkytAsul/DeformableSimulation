"""
First attempt to make a standalone program which displays
a MuJoCo scene on a VR device using openXR.

This attempt was made using ContextObject and eventually failed.
See why in the __init__ method.
"""
import mujoco.gl_context
import mujoco.glfw
import xr
from ctypes import byref

import mujoco

import glfw
from OpenGL import GL
import OpenGL.GL.ARB.framebuffer_object

class MuJoCoXr:
    def __init__(self):
        """
        Things preventing us from using an xr.ContextObject instead of creating everything
        ourselves:

        - CtxObj creates one swapchain per eye. We need only one stereo swapchain, because
          this is what the MuJoCo renderer is expecting.
            - This is bypassable by creating a new swapchain ourselves.
            - We also need to create a composition layer containing two views, one per eye.
            - And to be able to use the "ContextObject.graphics" pre-made functions, we also
              need another big layer with only one view containing both eyes side-by-side.
        - CtxObj uses an internal OpenGLGraphics containing a lot of useful functions related
          to rendering. However, this class initializes the OpenGL context using GLFW with some
          hints to require context version 4.5 and Core Profile. MuJoCo is not compatible with
          that: it requires OpenGL 3.1 or lower OR 3.2 and higher with the Compatibility Profile.
            - This is NOT bypassable without recompiling all of pyopenxr
        
        We must therefore do everything ourselves, just like in vanilla OpenXR.

        See the gl_example in the pyopenxr examples repository:
        https://github.com/cmbruns/pyopenxr_examples/blob/main/xr_examples/gl_example.py
        """
        self._context = xr.ContextObject(
            instance_create_info=xr.InstanceCreateInfo(
                enabled_extension_names=[
                    xr.KHR_OPENGL_ENABLE_EXTENSION_NAME
                ]
            )
        )

    def __enter__(self):
        self._context.__enter__()
        # the ContextObject does only create one swapchain per view (i.e. per eye)
        # whereas we want a "stereo" swapchain, as it is what MuJoCo OpenGL renderer is expecting

        gl_formats = xr.enumerate_swapchain_formats(self._context.session)
        print("Supported formats:", [hex(f) for f in gl_formats])

        gl_format = self._context.graphics.select_color_swapchain_format(gl_formats)
        print("Choosen", gl_format)

        self._fetch_view_attributes()

        self._swapchain = xr.create_swapchain(
            self._context.session,
            xr.SwapchainCreateInfo(
                array_size=1,
                format=gl_format,
                width=self._render_target_w,
                height=self._render_target_h,
                mip_count=1,
                face_count=1,
                sample_count=1,
                usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT,
            )
        )
        
        self._swapchain_images = xr.enumerate_swapchain_images(self._swapchain, xr.SwapchainImageOpenGLKHR)
        for i, si in enumerate(self._swapchain_images):
            print(f"Swapchain image {i} type = {xr.StructureType(si.type)}")

        self._render_layer = xr.CompositionLayerProjection(
            space=self._context.space,
            view_count=2,
            views=[xr.CompositionLayerProjectionView(
                sub_image=xr.SwapchainSubImage(
                    swapchain=self._swapchain,
                    image_rect=xr.Rect2Di(
                        extent=xr.Extent2Di(self._render_target_w // 2, self._render_target_h),
                        offset=xr.Offset2Di(x=i*self._render_target_w // 2) # so the right image is offset by its width
                    )
                )
            ) for i in range(2)]
        )
        self._large_layer_view = xr.CompositionLayerProjectionView(
            sub_image=xr.SwapchainSubImage(
                    swapchain=self._swapchain,
                    image_rect=xr.Rect2Di(
                        extent=xr.Extent2Di(self._render_target_w, self._render_target_h),
                    )
                )
        )
        # To use the already-made graphics class provided by openXR, we need a layer view that covers all the viewport.
        # It seems hacky, TODO see if it works.

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._context.__exit__(exc_type, exc_value, traceback)
    
    def _fetch_view_attributes(self):
        # we should normally get the render target size with the view configurations
        # but it has already been done by the ContextObject
        self._render_target_w = self._context.swapchains[0].width * 2
        self._render_target_h = self._context.swapchains[0].height
    
    def loop(self):
        for frame_state in self._context.frame_loop():
            print("Starting frame...")

            swapchain_index = xr.acquire_swapchain_image(self._swapchain, xr.SwapchainImageAcquireInfo())
            xr.wait_swapchain_image(self._swapchain, xr.SwapchainImageWaitInfo(xr.INFINITE_DURATION))
            swapchain_image = self._swapchain_images[swapchain_index]

            self._context.graphics.begin_frame(self._large_layer_view, swapchain_image.image)

            yield frame_state

            self._context.graphics.end_frame()
            xr.release_swapchain_image(self._swapchain, xr.SwapchainImageReleaseInfo())
            # TODO update pose here
            self._context.render_layers.append(byref(self._render_layer))
            print("End of frame.")

    def launch_mujoco(self, xml_path: str):
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        gl_context = mujoco.glfw.GLContext(1000, 600)
        gl_context.make_current()

        scene = mujoco.MjvScene(model, 1000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

        for frame_step in self.loop():
            mujoco.mj_step(model, data)
            
            mujoco.mjv_updateScene(model, data, None, None, None, mujoco.mjtCatBit.mjCAT_ALL, scene)
            viewport = mujoco.mjr_maxViewport(context)

            mujoco.mjr_render(viewport, scene, context)

if __name__ == "__main__":
    if False:
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        
        glfw.window_hint(glfw.DOUBLEBUFFER, False)
        # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
        # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        window = glfw.create_window(1000, 600, "gl_example", None, None)
        if window is None:
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(window)
        
        if not OpenGL.GL.ARB.framebuffer_object.glInitFramebufferObjectARB():
            raise RuntimeError("No arb fbo")

        for i in range(GL.glGetIntegerv(GL.GL_NUM_EXTENSIONS)):
            extension_name = GL.glGetStringi(GL.GL_EXTENSIONS, i)
            print("Supported OpenGL extension: ", extension_name)
    
    with MuJoCoXr() as mjxr:
        """for frame, frame_state in enumerate(mjxr.loop()):
        GL.glClearColor(1, 0.7, 0.7, 1)  # pink
        GL.glClear(GL.GL_COLOR_BUFFER_BIT) """
        mjxr.launch_mujoco("assets/MuJoCo scene.xml")