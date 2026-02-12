from .tele_style_node import (
    TeleStyleLoader,
    TeleStyleVideoInference,
    TeleStyleImageModelLoader,
    TeleStyleOfficialImageInference,
)

NODE_CLASS_MAPPINGS = {
    "TeleStyleLoader": TeleStyleLoader,
    "TeleStyleVideoInference": TeleStyleVideoInference,
    "TeleStyleImageModelLoader": TeleStyleImageModelLoader,
    "TeleStyleOfficialImageInference": TeleStyleOfficialImageInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TeleStyleLoader": "TeleStyle Model Loader",
    "TeleStyleVideoInference": "TeleStyle Video Transfer",
    "TeleStyleImageModelLoader": "TeleStyle Image Model Loader (Official)",
    "TeleStyleOfficialImageInference": "TeleStyle Image Transfer (Official)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
