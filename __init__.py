from .nodes.face_crop_node import NODE_CLASS_MAPPINGS as _crop_mappings
from .nodes.face_crop_node import NODE_DISPLAY_NAME_MAPPINGS as _crop_display
from .nodes.loop_nodes import NODE_CLASS_MAPPINGS as _loop_mappings
from .nodes.loop_nodes import NODE_DISPLAY_NAME_MAPPINGS as _loop_display

NODE_CLASS_MAPPINGS = {**_crop_mappings, **_loop_mappings}
NODE_DISPLAY_NAME_MAPPINGS = {**_crop_display, **_loop_display}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
