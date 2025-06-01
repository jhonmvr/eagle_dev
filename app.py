import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QFileDialog, QFrame
)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt

IMG_SIZE = 224
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']

def get_model_path():
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'covid_multitask_model.keras')
    return os.path.join(os.path.dirname(__file__), 'covid_multitask_model.keras')

model = tf.keras.models.load_model(get_model_path())

def predict_image(file_path):
    img = Image.open(file_path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    prediction = model.predict(img_arr)
    class_probs = prediction[0][0]
    predicted_class = CLASSES[np.argmax(class_probs)]
    confidence = np.max(class_probs)
    return predicted_class, confidence

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diagnóstico Pulmonar")
        self.setGeometry(200, 100, 500, 600)
        self.setStyleSheet("background-color: #ffffff; font-family: Arial;")

        self.layout = QVBoxLayout()
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(30, 30, 30, 30)

        # Título
        self.title = QLabel("🫁 Diagnóstico Pulmonar con IA")
        self.title.setFont(QFont("Arial", 18, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("color: black;")
        self.layout.addWidget(self.title)

        # Línea decorativa
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(line)

        # Imagen
        self.image_label = QLabel("Por favor, cargue una imagen de rayos X.")
        self.image_label.setFixedHeight(250)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: black; border: 2px dashed #cccccc;")

        self.layout.addWidget(self.image_label)

        # Resultado
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 14))
        self.result_label.setStyleSheet("color: #006699;")
        self.layout.addWidget(self.result_label)

        # Botón
        self.button = QPushButton("📂 Seleccionar imagen")
        self.button.setCursor(Qt.PointingHandCursor)
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #006699;
                color: white;
                padding: 12px;
                font-size: 14px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #005577;
            }
        """)
        self.button.clicked.connect(self.browse_file)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", "", "Imágenes (*.png *.jpg *.jpeg)"
        )
        if file_path:
            predicted_class, confidence = predict_image(file_path)
            self.result_label.setText(
                f"<b>Resultado:</b> {predicted_class}<br><b>Confianza:</b> {confidence:.2%}"
            )
            pixmap = QPixmap(file_path).scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
        else:
            self.result_label.setText("")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
