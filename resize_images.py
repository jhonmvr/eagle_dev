# resize_images.py

import os
import cv2

INPUT_ROOT = "data/COVID-19_Radiography_Dataset"
OUTPUT_ROOT = "output/resized"
TARGET_SIZE = (150, 150)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for class_name in os.listdir(INPUT_ROOT):
    class_path = os.path.join(INPUT_ROOT, class_name)
    images_folder = os.path.join(class_path, "images")
    
    if not os.path.isdir(images_folder):
        continue 

    output_class_dir = os.path.join(OUTPUT_ROOT, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    print(f"Procesando clase: {class_name}")

    for img_name in os.listdir(images_folder):
        input_img_path = os.path.join(images_folder, img_name)
        output_img_path = os.path.join(output_class_dir, img_name)

        try:
            image = cv2.imread(input_img_path)
            if image is not None:
                resized = cv2.resize(image, TARGET_SIZE)
                cv2.imwrite(output_img_path, resized)
        except Exception as e:
            print(f"Error al procesar {input_img_path}: {e}")
