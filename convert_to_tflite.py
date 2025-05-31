import tensorflow as tf

model = tf.keras.models.load_model("modelo_mobilenet.h5")

# Agrega firma explícita
input_shape = (1, 150, 150, 3)  # batch=1
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(input_shape, tf.float32)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

with open("mobilenet_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversión exitosa a mobilenet_model.tflite")
