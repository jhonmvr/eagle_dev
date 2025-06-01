
# Proyecto: Clasificación y Segmentación Pulmonar con MobileNetV2

## ⚙️ Configuración del entorno y ejecución distribuida

### 📁 Archivo `.env`

Guarda este archivo en la raíz del proyecto:

```env
# Configuraciones del modelo y entrenamiento
IMG_SIZE=224
BATCH_SIZE=8
EPOCHS=10
CLASSES=COVID,Normal,Viral Pneumonia

# Ruta del dataset en HDFS
DATASET_ROOT=hdfs:///user/jhon/dataset

# Archivos de salida
MODEL_PATH=covid_multitask_model.keras
TRAINING_LOG=training_log.csv
```

> ✅ Usa [`python-dotenv`](https://pypi.org/project/python-dotenv/) para cargar automáticamente estas variables en el script de entrenamiento.

---

### 🧠 Entrenamiento distribuido (opcional)

Para ejecutar el entrenamiento de forma paralela en múltiples nodos con `MultiWorkerMirroredStrategy`, necesitas definir la variable `TF_CONFIG` en cada nodo. Ejemplo para 2 nodos locales:

#### 🖥️ Nodo 0:

```bash
export TF_CONFIG='{
  "cluster": {
    "worker": ["localhost:12345", "localhost:12346"]
  },
  "task": {
    "type": "worker",
    "index": 0
}'
python train_with_hadoop.py
```

#### 🖥️ Nodo 1:

```bash
export TF_CONFIG='{
  "cluster": {
    "worker": ["localhost:12345", "localhost:12346"]
  },
  "task": {
    "type": "worker",
    "index": 1
}'
python train_with_hadoop.py
```

Puedes adaptar `"localhost:12345"` a direcciones IP reales si estás en red.

---

### 📦 Dependencias requeridas

```bash
pip install tensorflow pandas pillow matplotlib python-dotenv
```

---

### 🔎 Consejos de ejecución

- Asegúrate de que los archivos `.metadata.xlsx`, imágenes y máscaras estén disponibles en el mismo HDFS que especificaste en `DATASET_ROOT`.
- Si no usas múltiples nodos, el entrenamiento funcionará en modo normal sin necesidad de definir `TF_CONFIG`.
- El modelo y los logs se guardan automáticamente con los nombres definidos en el `.env`.

