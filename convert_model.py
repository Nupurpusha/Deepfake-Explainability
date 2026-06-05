import tensorflow as tf
from tensorflow.keras.models import model_from_json
import h5py
import json

# Open old model
with h5py.File("best_deepfake_detector_model.h5", "r") as f:
    model_config = f.attrs.get("model_config")

    if model_config is None:
        raise ValueError("No model config found.")

    model_config = json.loads(model_config)

# Fix layers
for layer in model_config["config"]["layers"]:

    config = layer["config"]

    # Fix InputLayer
    if layer["class_name"] == "InputLayer":
        if "batch_shape" in config:
            config["batch_input_shape"] = config.pop("batch_shape")

    # Fix dtype policy
    if "dtype" in config and isinstance(config["dtype"], dict):
        config["dtype"] = "float32"

# Rebuild model
model = model_from_json(json.dumps(model_config))

# Load weights
model.load_weights("best_deepfake_detector_model.h5")

# Save compatible model
model.save("fixed_model.h5")

print("Model fixed successfully!")