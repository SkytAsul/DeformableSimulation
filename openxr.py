from collections.abc import Callable
from OpenGL import GL
import xr

class OpenXrConnector:
    def __init__(self):
        self._context = xr.ContextObject(
            instance_create_info=xr.InstanceCreateInfo(
                enabled_extension_names=[
                    xr.KHR_OPENGL_ENABLE_EXTENSION_NAME
                ]
            )
        )

    def __enter__(self):
        self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._context.__exit__(exc_type, exc_value, traceback)

    def main_loop(self):
        for frame_index, self._frame_state in enumerate(self._context.frame_loop()):
            yield frame_index

    def get_eyes_poses(self):
        view_state, views = xr.locate_views(self._context.session,
            view_locate_info=xr.ViewLocateInfo(
                view_configuration_type=self._context.view_configuration_type,
                display_time=self._frame_state.predicted_display_time,
                space=self._context.space,
            ))
        return views

    def draw_pink(self):
        for frame_index, frame_state in enumerate(self._context.frame_loop()):
            for view in self._context.view_loop(frame_state):
                GL.glClearColor(1, 0.7, 0.7, 1)  # pink
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            if frame_index > 500:  # Don't run forever
                break


if __name__ == "__main__":
    with OpenXrConnector() as openxr:
        openxr.main()