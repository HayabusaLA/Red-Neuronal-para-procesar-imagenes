import os, json, glob, random, math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from PIL import Image

# -------- Config --------
DATA_ROOT = os.path.join("data", "ncats", "dataset")
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42
OUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUT_DIR, "models", "catvsnocat.keras")
HIST_PATH = os.path.join(OUT_DIR, "history", "history.json")

os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "history"), exist_ok=True)

# ====== UTILIDADES COMUNES ======
def build_model(input_shape=(64,64,3), num_classes=2):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ====== MODO A: CSVs (train/test) ======
def have_csv_layout(root):
    tdir = os.path.join(root, "train")
    sdir = os.path.join(root, "test")
    return all(os.path.isfile(os.path.join(tdir, f)) for f in ["train_images.csv", "train_labels.csv"]) and \
           all(os.path.isfile(os.path.join(sdir, f)) for f in ["test_images.csv", "test_labels.csv"])

def _infer_img_shape(num_features):
    # Probamos RGB (3 canales) y gris (1 canal) con forma cuadrada
    for ch in (3, 1):
        n2 = num_features / ch
        n = int(round(math.sqrt(n2)))
        if n * n * ch == num_features:
            return (n, n, ch)
    # fallback típico de la tarea: 64x64x3 o 64x64x1
    if num_features == 12288:
        return (64, 64, 3)
    if num_features == 4096:
        return (64, 64, 1)
    raise ValueError(f"No puedo inferir forma cuadrada a partir de {num_features} features.")

def _load_csv_pair(img_csv, lbl_csv):
    # Tolerante a encabezados: usamos numpy genfromtxt que ignora strings
    X = np.genfromtxt(img_csv, delimiter=",")
    y = np.genfromtxt(lbl_csv, delimiter=",")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if y.ndim > 1:
        y = y.reshape(-1)
    # inferir forma
    h, w, c = _infer_img_shape(X.shape[1])
    X = X.astype(np.float32) / 255.0
    X = X.reshape((-1, h, w, c))
    # si es 1 canal, duplicamos a 3 para la CNN estándar
    if c == 1:
        X = np.repeat(X, 3, axis=-1)
        c = 3
    # normalizamos etiquetas a enteros consecutivos
    uniq = np.unique(y)
    mapping = {val: idx for idx, val in enumerate(sorted(uniq))}
    y = np.array([mapping[val] for val in y], dtype=np.int64)
    class_names = [str(k) for k in sorted(uniq)]  # p.ej. "0","1"
    # si sabemos que 0/1 significa cat/nocat, puedes renombrar:
    if len(class_names) == 2 and set(class_names) == {"0", "1"}:
        class_names = ["cat", "nocat"]  # opcional
    return X, y, class_names, (h, w, 3)

def load_from_csv_layout(root):
    train_dir = os.path.join(root, "train")
    test_dir  = os.path.join(root, "test")
    X_tr, y_tr, class_names, ishape = _load_csv_pair(
        os.path.join(train_dir, "train_images.csv"),
        os.path.join(train_dir, "train_labels.csv"),
    )
    X_te, y_te, _, _ = _load_csv_pair(
        os.path.join(test_dir, "test_images.csv"),
        os.path.join(test_dir, "test_labels.csv"),
    )
    return (X_tr, y_tr, X_te, y_te, class_names, ishape)

# ====== MODO B: Carpetas de imágenes (cat/nocat) ======
def load_images_from_folders(folder_map, img_size=(64,64)):
    from PIL import Image
    X, y, class_names = [], [], sorted(folder_map.keys())
    for label, folder in folder_map.items():
        files = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            files.extend(glob.glob(os.path.join(folder, ext)))
        for f in files:
            try:
                img = Image.open(f).convert("RGB").resize(img_size)
                X.append(np.asarray(img, dtype=np.float32)/255.0)
                y.append(class_names.index(label))
            except Exception:
                continue
    X = np.stack(X, axis=0) if len(X) else np.zeros((0, img_size[0], img_size[1], 3), dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, class_names

def infer_folders_layout(data_root):
    a_train_cat = os.path.join(data_root, "train", "cat")
    a_train_nocat = os.path.join(data_root, "train", "nocat")
    a_test_cat  = os.path.join(data_root, "test", "cat")
    a_test_nocat= os.path.join(data_root, "test", "nocat")
    if all(os.path.isdir(p) for p in [a_train_cat, a_train_nocat, a_test_cat, a_test_nocat]):
        return (
            {"cat": a_train_cat, "nocat": a_train_nocat},
            {"cat": a_test_cat, "nocat": a_test_nocat},
        )
    b_cat = os.path.join(data_root, "cat")
    b_nocat = os.path.join(data_root, "nocat")
    if all(os.path.isdir(p) for p in [b_cat, b_nocat]):
        return ({"cat": b_cat, "nocat": b_nocat}, None)
    # búsqueda recursiva
    found_cat, found_nocat = [], []
    for root, dirs, files in os.walk(data_root):
        if os.path.basename(root).lower() == "cat":
            found_cat.append(root)
        if os.path.basename(root).lower() == "nocat":
            found_nocat.append(root)
    for c in found_cat:
        parent = os.path.dirname(c)
        if os.path.join(parent, "nocat") in found_nocat:
            return ({"cat": c, "nocat": os.path.join(parent, "nocat")}, None)
    return None

# ====== MAIN ======
if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    if have_csv_layout(DATA_ROOT):
        X_tr, y_tr, X_te, y_te, class_names, ishape = load_from_csv_layout(DATA_ROOT)
        input_shape = ishape  # ya forzado a 3 canales
    else:
        layouts = infer_folders_layout(DATA_ROOT)
        if layouts is None:
            raise RuntimeError(f"No se encontró estructura válida en {DATA_ROOT}")
        train_map, test_map = layouts
        # tamaño por defecto si son imágenes
        IMG_SIZE = (64, 64)
        if test_map is None:
            X, y, class_names = load_images_from_folders(train_map, IMG_SIZE)
            if X.shape[0] == 0:
                raise RuntimeError("No se cargaron imágenes. Verifica rutas del dataset.")
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
        else:
            X_tr, y_tr, class_names = load_images_from_folders(train_map, IMG_SIZE)
            X_te, y_te, _          = load_images_from_folders(test_map, IMG_SIZE)
        input_shape = (64,64,3)

    model = build_model(input_shape=input_shape, num_classes=len(np.unique(y_tr)))
    history = model.fit(X_tr, y_tr, validation_split=0.15, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    eval_res = model.evaluate(X_te, y_te, verbose=0)
    print({"test_loss": float(eval_res[0]), "test_acc": float(eval_res[1])})

    model.save(MODEL_PATH)
    with open(HIST_PATH, "w") as f:
        json.dump({"history": {k: list(v) for k, v in history.history.items()},
                   "test": {"loss": float(eval_res[0]), "acc": float(eval_res[1])},
                   "classes": list(map(str, class_names))}, f, indent=2)

    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Historial guardado en: {HIST_PATH}")
