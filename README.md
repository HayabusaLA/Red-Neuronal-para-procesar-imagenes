# CNN Cat vs No Cat (Keras)

**Propósito:** Aplicar una red neuronal convolutiva al procesamiento de imágenes para clasificar *cat* vs *no cat*.

## Dataset
Usa el dataset del profesor: `adsoftsito/ncats`. Clona el repo dentro de `data/` así:
```bash
cd data
git clone https://github.com/adsoftsito/ncats.git
cd ..
```

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
- Accuracy (test): _(pega aquí el valor impreso en consola)_
- Evidencia: agrega screenshots de la carpeta de predicciones.

## Autor
- Tu nombre — Matrícula
- Curso / Profesor
