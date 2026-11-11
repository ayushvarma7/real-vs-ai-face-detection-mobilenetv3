import os
import numpy as np
from PIL import Image
import keras

model = keras.models.load_model("best_model.keras")
print("Input shape:", model.input_shape)


correct, total = 0, 0

for label, folder in [(1, "test_images/real"), (0, "test_images/ai" )]:
    for f in sorted(os.listdir(folder)):
        img = Image.open(os.path.join(folder, f)).convert("RGB").resize((128,128))
        x=np.array(img, dtype="float32")[None, ...]
        p = float(model.predict(x, verbose=0)[0,0])
        correct+= int((p>=0.5)==label)
        total+=1

print(f"Accuracy on {total} game images: {correct/total:.2f}")