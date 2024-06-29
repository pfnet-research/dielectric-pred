import os
import pathlib


def model_directory() -> pathlib.Path:
    if "PFP_MODEL_PATH" in os.environ:
        model_dir = pathlib.Path(os.environ["PFP_MODEL_PATH"])
    else:
        model_dir = pathlib.Path(__file__).parent / "configs"
    return model_dir


DEFAULT_MODEL_DIRECTORY = model_directory()
DEFAULT_MODEL = DEFAULT_MODEL_DIRECTORY / "model_v1_4_1.yaml"
EDGE_FULL_MODEL = DEFAULT_MODEL_DIRECTORY / "edge_full_model.yaml"
