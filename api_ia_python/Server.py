import cv2
import numpy as np
from keras.models import model_from_json
from flask import Flask, request, jsonify
import base64
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
# emotion_dict = {0: "Enojado", 1: "Disgustado", 2: "Disgustado", 3: "Feliz", 4: "Neutral", 5: "Triste", 6: "Sorprendido"}

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    image_data = request.json['image']
    image_data = base64.b64decode(image_data)

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'error': 'No se detectó ninguna cara en la imagen.'}), 400

    emotions = []

    for (x, y, w, h) in faces:
        roi_gray = gray_img[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        return jsonify({'emotion': emotion_dict[maxindex]}), 200

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 3001), app)
    http_server.serve_forever()
