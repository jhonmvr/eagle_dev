import pandas as pd
import matplotlib.pyplot as plt

# Leer el CSV
log_df = pd.read_csv('training_log.csv')

# Convertir timestamps a datetime
log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])

# Calcular duración por época en segundos
log_df['epoch_time_sec'] = log_df['timestamp'].diff().dt.total_seconds()
log_df['epoch_time_sec'].fillna(0, inplace=True)

# ========== MÉTRICAS DEL MODELO ==========

# Accuracy clasificación
plt.figure()
plt.plot(log_df['classification_accuracy'], label='Train Accuracy')
plt.plot(log_df['val_classification_accuracy'], label='Val Accuracy')
plt.title('Precisión de Clasificación')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')

# Pérdida total
plt.figure()
plt.plot(log_df['loss'], label='Train Loss')
plt.plot(log_df['val_loss'], label='Val Loss')
plt.title('Pérdida Total')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')

# Pérdida de segmentación
plt.figure()
plt.plot(log_df['segmentation_loss'], label='Train Segmentation Loss')
plt.plot(log_df['val_segmentation_loss'], label='Val Segmentation Loss')
plt.title('Pérdida de Segmentación')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('segmentation_loss_plot.png')

# ========== MÉTRICAS DEL SISTEMA ==========

# CPU
plt.figure()
plt.plot(log_df['cpu_percent'], label='CPU (%)')
plt.title('Uso de CPU durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('CPU %')
plt.legend()
plt.grid(True)
plt.savefig('cpu_usage.png')

# RAM
plt.figure()
plt.plot(log_df['ram_percent'], label='RAM (%)', color='orange')
plt.title('Uso de RAM durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('RAM %')
plt.legend()
plt.grid(True)
plt.savefig('ram_usage.png')

# GPU
plt.figure()
plt.plot(log_df['gpu_percent'], label='GPU (%)', color='green')
plt.title('Uso de GPU durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('GPU %')
plt.legend()
plt.grid(True)
plt.savefig('gpu_usage.png')

# Memoria de GPU
plt.figure()
plt.plot(log_df['gpu_memory_percent'], label='GPU Memory (%)', color='red')
plt.title('Uso de Memoria de GPU durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('GPU Memory %')
plt.legend()
plt.grid(True)
plt.savefig('gpu_memory_usage.png')

# Tiempo por época 
log_df['epoch_time_min'] = log_df['epoch_time_sec'] / 60

plt.figure()
plt.plot(log_df['epoch_time_min'], label='Duración por época (min)', marker='o')
plt.title('Tiempo de procesamiento por época')
plt.xlabel('Época')
plt.ylabel('Minutos')
plt.legend()
plt.grid(True)
plt.savefig('epoch_time_plot.png')

# Imprimir resumen
print("✅ Gráficas guardadas:")
print("- accuracy_plot.png")
print("- loss_plot.png")
print("- segmentation_loss_plot.png")
print("- cpu_usage.png")
print("- ram_usage.png")
print("- gpu_usage.png")
print("- gpu_memory_usage.png")
print("- epoch_time_plot.png")
print("✅ Archivo CSV actualizado: training_log_con_tiempo.csv")
