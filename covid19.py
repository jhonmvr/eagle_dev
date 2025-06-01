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
from tensorflow.keras.callbacks import CSVLogger


csv_logger = CSVLogger('training_log.csv', append=False)


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