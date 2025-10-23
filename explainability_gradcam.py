import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import BinaryScore

# --- CONFIGURATION ---
MODEL_PATH = "models/inceptionv3_fold1.keras"      # <- Update to saved model, per fold
EXAMPLE_SPEC = "data/spectrograms_227x227_rgb/s60w.npy"  # <- One sample to explain

# 1. Load trained model --
assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Find last convolutional layer
conv_names = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
if not conv_names:
    raise RuntimeError("No Conv2D layers found in model. Grad-CAM requires a Conv2D-based model.")
last_conv_layer = conv_names[-1]
print("Using Grad-CAM on conv layer:", last_conv_layer)

# 3. Load and preprocess a spectrogram sample
assert os.path.exists(EXAMPLE_SPEC), f"Spectrogram file not found: {EXAMPLE_SPEC}"
img = np.load(EXAMPLE_SPEC)  # shape should be (227, 227, 3)
if img.max() > 1.01:
    img = img / 255.0
input_image = np.expand_dims(img, axis=0)

# 4. Set up Grad-CAM
gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
score = BinaryScore(1)          # For binary classifier, 1 = "SZ"/positive class

# 5. Generate Grad-CAM heatmap
cam = gradcam(score, input_image, penultimate_layer=last_conv_layer)
heatmap = cam[0]

# 6. Overlay heatmap on spectrogram
plt.figure(figsize=(6, 6))
plt.imshow(img)                            # spectrogram in background
plt.imshow(heatmap, cmap='jet', alpha=0.5) # Grad-CAM overlay
plt.title('Grad-CAM Visualization')
plt.axis('off')
plt.tight_layout()

os.makedirs("gradcam_outputs", exist_ok=True)
plt.savefig('gradcam_outputs/gradcam_sample.png', dpi=200)
plt.show()
