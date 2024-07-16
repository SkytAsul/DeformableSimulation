#import "@preview/codelst:2.0.1": sourcecode, sourcefile

#sourcecode(gobble: auto)[```py
        self._xr_system = xr.get_system(self._xr_instance, xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY))
        assert xr.enumerate_view_configurations(self._xr_instance, self._xr_system)[0] == xr.ViewConfigurationType.PRIMARY_STEREO
        
        views_config = xr.enumerate_view_configuration_views(self._xr_instance, self._xr_system, xr.ViewConfigurationType.PRIMARY_STEREO)
        assert len(views_config) == 2
        assert views_config[0].recommended_image_rect_width == views_config[1].recommended_image_rect_width
        assert views_config[0].recommended_image_rect_height == views_config[1].recommended_image_rect_height
        
        self._width, self._height = views_config[0].recommended_image_rect_width, views_config[0].recommended_image_rect_height
        self._width_render = self._width * 2
```]