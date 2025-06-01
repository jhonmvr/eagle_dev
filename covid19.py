import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Conv2D, UpSampling2D
from keras.saving import save_model
from tensorflow.keras.callbacks import Callback
from datetime import datetime
import csv
import os
import json
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class ResourceLoggingCSVLogger(Callback):
    def __init__(self, csv_filename='training_log.csv', jsonl_filename='epoch_logs.jsonl', append=False):
        super().__init__()
        self.csv_filename = csv_filename
        self.jsonl_filename = jsonl_filename
        self.append = append
        self.writer = None
        self.csv_file = None
        self.keys = None

    def on_train_begin(self, logs=None):
        mode = 'a' if self.append else 'w'
        self.csv_file = open(self.csv_filename, mode, newline='')
        self.writer = None

        # Iniciar o limpiar JSONL
        if not self.append and os.path.exists(self.jsonl_filename):
            os.remove(self.jsonl_filename)

    def get_resource_usage(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        gpu_load, gpu_mem = (None, None)
        if GPU_AVAILABLE:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_load = gpus[0].load * 100
                gpu_mem = gpus[0].memoryUtil * 100
        return cpu, mem, gpu_load, gpu_mem

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        cpu, mem, gpu, gpu_mem = self.get_resource_usage()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        full_logs = {
            'epoch': epoch,
            'timestamp': timestamp,
            'cpu_percent': cpu,
            'ram_percent': mem,
            'gpu_percent': gpu if gpu is not None else 'N/A',
            'gpu_memory_percent': gpu_mem if gpu_mem is not None else 'N/A',
        }
        full_logs.update(logs)

        # --- CSV principal ---
        if self.writer is None:
            self.keys = list(full_logs.keys())
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.keys)
            if not self.append or os.stat(self.csv_filename).st_size == 0:
                self.writer.writeheader()

        self.writer.writerow(full_logs)
        self.csv_file.flush()

        # --- JSONL (detalle de cada época) ---
        with open(self.jsonl_filename, 'a') as jf:
            json.dump(full_logs, jf)
            jf.write('\n')

    def on_train_end(self, logs=None):
        self.csv_file.close()


csv_logger = ResourceLoggingCSVLogger(
    csv_filename='training_log.csv',
    jsonl_filename='epoch_logs.jsonl',
    append=False
)


# CONFIG
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}
DATASET_ROOT = 'dataset'

# DATA GENERATOR
class MultiTaskDataGenerator(Sequence):
    def __init__(self, df, batch_size=8, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_idxs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idxs]

        images, masks, labels = [], [], []

        for _, row in batch_df.iterrows():
            image_path = row['image_path']
            mask_path = row['mask_path']
            label = row['label']
            try:
                img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
                mask = Image.open(mask_path).resize((IMG_SIZE, IMG_SIZE)).convert('L')
                img_arr = np.array(img, dtype=np.float32) / 255.0
                mask_arr = np.expand_dims(np.array(mask, dtype=np.float32) / 255.0, axis=-1)
                images.append(img_arr)
                masks.append(mask_arr)
                labels.append(tf.keras.utils.to_categorical(label, num_classes=len(CLASSES)))
            except Exception as e:
                print(f"Error cargando {image_path} o {mask_path}: {e}")

        return np.array(images), {
            'classification': np.array(labels),
            'segmentation': np.array(masks)
        }

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# METADATA
def load_metadata(dataset_root="dataset"):
    all_data = []
    for class_name in CLASSES:
        meta_file = os.path.join(dataset_root, f"{class_name}.metadata.xlsx")
        df = pd.read_excel(meta_file)
        for _, row in df.iterrows():
            filename = f"{row['FILE NAME']}.png"
            image_path = os.path.join(dataset_root, class_name, 'images', filename)
            mask_path = os.path.join(dataset_root, class_name, 'masks', filename)
            if os.path.exists(image_path) and os.path.exists(mask_path):
                all_data.append({
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'label': CLASS_TO_INDEX[class_name]
                })
    return pd.DataFrame(all_data)

# MODEL
def build_model():
    base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs = base.input
    x = base.output

    # Clasificación
    x_cls = GlobalAveragePooling2D()(x)
    x_cls = Dropout(0.3)(x_cls)
    out_class = Dense(3, activation='softmax', name='classification')(x_cls)

    # Segmentación (decoder completo hasta 224x224)
    x_seg = UpSampling2D(size=2)(x)        # 7 → 14
    x_seg = Conv2D(64, 3, padding='same', activation='relu')(x_seg)
    x_seg = UpSampling2D(size=2)(x_seg)    # 14 → 28
    x_seg = Conv2D(32, 3, padding='same', activation='relu')(x_seg)
    x_seg = UpSampling2D(size=2)(x_seg)    # 28 → 56
    x_seg = Conv2D(16, 3, padding='same', activation='relu')(x_seg)
    x_seg = UpSampling2D(size=4)(x_seg)    # 56 → 224
    out_mask = Conv2D(1, 1, activation='sigmoid', name='segmentation')(x_seg)

    return Model(inputs=inputs, outputs=[out_class, out_mask])

# MAIN PIPELINE
df_all = load_metadata(DATASET_ROOT)
df_train, df_val = train_test_split(df_all, test_size=0.2, random_state=42)
train_gen = MultiTaskDataGenerator(df_train, batch_size=BATCH_SIZE, shuffle=True)
val_gen = MultiTaskDataGenerator(df_val, batch_size=BATCH_SIZE, shuffle=False)

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        'classification': 'categorical_crossentropy',
        'segmentation': 'binary_crossentropy'
    },
    metrics={
        'classification': 'accuracy',
        'segmentation': 'mse'
    }
)

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[csv_logger])
model.save("covid_multitask_model.h5")
save_model(model, "covid_multitask_model.keras")