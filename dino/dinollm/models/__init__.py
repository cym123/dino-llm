
from .config import ModelConfig, RotaryConfig
from .register import get_model_class
from .weight import load_weight


def create_model(model_config: ModelConfig):
    return get_model_class(model_config.architectures[0], model_config)


__all__ = ["create_model", "load_weight", "RotaryConfig"]
