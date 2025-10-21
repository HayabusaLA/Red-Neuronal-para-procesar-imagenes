import os, glob, shutil
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = os.path.join("outputs", "models", "catvsnocat.keras")
IMG_SIZE = (64, 64)

IN_DIRS = [
    os.path.join("data", "ncats", "predict", "cat"),
    os.path.join("data", "ncats", "predict", "nocat"),
]

OUT_BASE = os.path.join("outputs", "predictions")
OUT_CAT = os.path.join(OUT_BASE, "cat")
OUT_NOCAT = os.path.join(OUT_BASE, "nocat")
os.makedirs(OUT_CAT, exist_ok=True)
os.makedirs(OUT_NOCAT, exist_ok=True)

def load_img(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)/255.0
    return np.expand_dims(arr, axis=0)

if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH)
    files = []
    for d in IN_DIRS:
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            files.extend(glob.glob(os.path.join(d, ext)))

    for f in files:
        x = load_img(f)
        p = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(p))
        conf = float(np.max(p))
        # Por convención, asumimos class_names ordenadas alfabeticamente: ["cat","nocat"]
        target_dir = OUT_CAT if pred_idx == 0 else OUT_NOCAT
        base = os.path.basename(f)
        out_name = f"{os.path.splitext(base)[0]}__{'cat' if pred_idx==0 else 'nocat'}__{conf:.2f}{os.path.splitext(base)[1]}"
        shutil.copyfile(f, os.path.join(target_dir, out_name))
    print(f"Listo. Revisa {OUT_CAT} y {OUT_NOCAT}")
