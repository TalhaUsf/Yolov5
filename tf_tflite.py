#%%
from pathlib import Path
import tensorflow as tf
from rich.console import Console

saved_model_path = Path("weights/yolov5")
#%%
model = tf.saved_model.load(saved_model_path.as_posix())
Console().print(f"⚡\t model {model.signatures.keys()}")

# %%
