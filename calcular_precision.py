import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# === Leer todos los CSV generados por Spark ===
csv_files = glob.glob("predicciones.csv/part-*.csv")
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Normalizar nombres de columnas (opcional)
df.columns = [col.strip().capitalize() for col in df.columns]

# === Precisión general ===
accuracy = accuracy_score(df["Actual"], df["Predicted"])
print(f"\n✅ Precisión total: {accuracy * 100:.2f}%")

# === Tabla de conteo por clase ===
conteo = df.groupby(["Actual", "Predicted"]).size().unstack(fill_value=0)
print("\n📊 Conteo por clase (tabla de clasificación):")
print(conteo)

# === Matriz de confusión ===
labels = sorted(df["Actual"].unique())
cm = confusion_matrix(df["Actual"], df["Predicted"], labels=labels)

# === Mostrar y guardar matriz de confusión ===
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.savefig("matriz_confusion.png")
plt.show()

# === Precisión por clase (recall) ===
aciertos_por_clase = conteo.lookup(labels, labels)
total_por_clase = conteo.sum(axis=1).values
precision_por_clase = (aciertos_por_clase / total_por_clase) * 100

# === Gráfico de precisión por clase ===
plt.figure()
plt.bar(labels, precision_por_clase)
plt.ylabel("Precisión (%)")
plt.title("Precisión por clase")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("precision_por_clase.png")
plt.show()
