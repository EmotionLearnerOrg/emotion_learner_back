import cv2
import numpy as np
from keras.models import model_from_json
from flask import Flask, request, jsonify
from PIL import Image
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

# Cargar modelos de emociones
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
print("Modelo cargado correctamente")

emotion_dict = {0: "Enojo", 1: "Tristeza", 2: "Tristeza", 3: "Alegria", 4: "Calma", 5: "Tristeza", 6: "Sorpresa"}

@app.route('/detect-emotion', methods=['POST'])
def detect_emotions():
    
    images = request.files.getlist('images')
    if not images:
        return jsonify({'error': 'No se han enviado imágenes'}), 400

    results = {}
    for index, image_file in enumerate(images):
        img = Image.open(image_file)
        gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            results[f'emotion_{index + 1}'] = 'No se detectó ninguna cara en la imagen.'
        else:
            emotions = []

            for (x, y, w, h) in faces:
                roi_gray = gray_img[y:y+h, x:x+w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                emotions.append(emotion_dict[maxindex])
            
            # Encuentra la emoción más común en la imagen
            most_common_emotion = max(set(emotions), key=emotions.count)
            results[f'emotion_{index + 1}'] = most_common_emotion

    return jsonify(results), 200

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 3001), app)
    http_server.serve_forever()