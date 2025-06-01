import pandas as pd
import matplotlib.pyplot as plt

# Leer CSV generado por CSVLogger
log_df = pd.read_csv('training_log.csv')

# Graficar precisión de clasificación
plt.figure()
plt.plot(log_df['classification_accuracy'], label='train_accuracy')
plt.plot(log_df['val_classification_accuracy'], label='val_accuracy')
plt.title('Accuracy (Classification)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')

# Graficar pérdida total
plt.figure()
plt.plot(log_df['loss'], label='train_loss')
plt.plot(log_df['val_loss'], label='val_loss')
plt.title('Loss (Total)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')

# Graficar pérdida de segmentación
plt.figure()
plt.plot(log_df['segmentation_loss'], label='train_seg_loss')
plt.plot(log_df['val_segmentation_loss'], label='val_seg_loss')
plt.title('Loss (Segmentation)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('segmentation_loss_plot.png')
