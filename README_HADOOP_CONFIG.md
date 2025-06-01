
# Proyecto: Clasificación y Segmentación Pulmonar con MobileNetV2

## ⚙️ Configuración del entorno y ejecución distribuida

### 📁 Archivo `.env`

Guarda este archivo en la raíz del proyecto:

```env
# Configuraciones del modelo y entrenamiento
IMG_SIZE=224
BATCH_SIZE=8
EPOCHS=10
CLASSES=COVID,Normal,Viral_Pneumonia

# Ruta del dataset en HDFS
DATASET_ROOT=hdfs:///user/jhon/dataset

# Archivos de salida
MODEL_PATH=covid_multitask_model.keras
TRAINING_LOG=training_log.csv
```

> ✅ Usa [`python-dotenv`](https://pypi.org/project/python-dotenv/) para cargar automáticamente estas variables en el script de entrenamiento.

---

### 🚀 Acceso a la Interfaz Gráfica de Hadoop (NameNode)

Con tu configuración Docker, puedes acceder a la interfaz gráfica de HDFS en:

```
http://localhost:9870
```

Desde allí puedes explorar los archivos del HDFS:
- Utiliza **“Utilities > Browse the file system”**
- O entra directo: `http://localhost:9870/explorer.html#/user/jhon/dataset`

---

### 🧠 Entrenamiento distribuido (opcional)

Para ejecutar el entrenamiento de forma paralela en múltiples nodos con `MultiWorkerMirroredStrategy`, necesitas definir la variable `TF_CONFIG` en cada nodo. Ejemplo para 2 workers:

#### 🖥️ Nodo 0:

```bash
export TF_CONFIG='{
  "cluster": {
    "worker": ["localhost:12345", "localhost:12346"]
  },
  "task": {
    "type": "worker",
    "index": 0
  }
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
  }
}'
python train_with_hadoop.py
```

Puedes reemplazar `"localhost"` por la IP real del contenedor o nodo en red.

---

### 🐳 Puertos definidos en Docker (`docker-compose.yml`)

Asegúrate de tener expuestos estos puertos en tu archivo:

```yaml
namenode:
  ports:
    - "9870:9870"  # Web UI del NameNode
    - "9000:9000"  # RPC principal del HDFS
```

---

### 📦 Dependencias requeridas

```bash
pip install tensorflow pandas pillow matplotlib python-dotenv
```

---

### 🔎 Consejos de ejecución

- Verifica que los archivos `.metadata.xlsx`, imágenes y máscaras estén correctamente subidos a HDFS.
- Usa `hdfs dfs -ls /user/jhon/dataset` para confirmar.
- El modelo y los logs se guardan automáticamente con los nombres definidos en el `.env`.
- En producción o entrenamiento prolongado, asegúrate de monitorear el uso de RAM y CPU.

---
