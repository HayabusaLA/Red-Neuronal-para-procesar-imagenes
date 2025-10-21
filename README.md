# CNN Cat vs No Cat (Keras)
## Autor
- Luis Adrian Abarca Gomez — A01798043

  
**Propósito:** Aplicar una red neuronal convolutiva al procesamiento de imágenes para clasificar *cat* vs *no cat*.

## Reproducibilidad
```bash
python -m venv .venv
# Activa el entorno:
#   Windows: .venv\Scripts\activate
#   macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

# Clona el dataset
cd data
git clone https://github.com/adsoftsito/ncats.git
cd ..

# Entrena y evalúa
python train.py

# Predice y separa imágenes en outputs/predictions/{cat,nocat}
python predict.py
```

## Artefactos
- `outputs/models/catvsnocat.keras`
- `outputs/history/history.json`
- `outputs/predictions/{cat,nocat}/`

## Resultados
```bash
Test Accuracy: 0.88
Test Loss: 0.47
Épocas: 15
Optimizer: Adam (lr = 0.001)
Loss Function: Sparse Categorical Crossentropy
Dataset usado: ncats (cat vs nocat)

El modelo final se guardó en:
outputs/models/catvsnocat.keras

El historial de métricas se guardó en:
outputs/history/history.json
```


  
