import tensorflow as tf
from keras.models import model_from_json

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

print(" Convierte el modelo a TensorFlow Lite")
converter = tf.lite.TFLiteConverter.from_keras_model(emotion_model)
print(" # Aplica la cuantización")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

print(" Guarda el modelo convertido en un archivo .tflite")
with open('output_model.tflite', 'wb') as f:
    f.write(tflite_model)