# spark_job.py

from pyspark.sql import SparkSession
import tensorflow as tf
import numpy as np
import cv2
import os

# === Configuración ===
MODEL_PATH = "mobilenet_model.tflite"
IMAGE_DIR = "output/resized"
IMG_SIZE = (150, 150)

# === Clases ===
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# === Inicializa Spark ===
spark = SparkSession.builder \
    .appName("ImageClassifier") \
    .master("local[*]") \
    .getOrCreate()

# === Cargar modelo TFLite ===
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# === Función de predicción ===
def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return -1
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    return int(np.argmax(pred))

# === Recolectar datos ===
results = []

for class_name in os.listdir(IMAGE_DIR):
    class_path = os.path.join(IMAGE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        pred_class = predict_image(img_path)
        results.append((img_name, class_name, CLASSES[pred_class] if pred_class != -1 else "Error"))

# === Crear y mostrar DataFrame ===
df = spark.createDataFrame(results, ["Image", "Actual", "Predicted"])
df.show(20, truncate=False)

# === Guardar resultados (opcional) ===
df.write.csv("predicciones.csv", header=True, mode="overwrite")

spark.stop()
