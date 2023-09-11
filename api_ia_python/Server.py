import cv2
import numpy as np
from keras.models import model_from_json
from flask import Flask, request, jsonify
from PIL import Image
from gevent.pywsgi import WSGIServer
import os

app = Flask(__name__)

# A modo de testing, guardo las imagenes procesadas por la api (Descomentarlo) -> Lineas 13,14,15,54,55,56,77,78,79
# UPLOAD_FOLDER = 'D:/Hola Mundo/Proyectos/mi_mundo_emocional/mi_mundo_emocional_back/imagenes_procesadas'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar modelos de emociones
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
print("Modelo cargado correctamente")

emotion_dict = {0: "Enojo", 1: "Tristeza", 2: "Tristeza",
                3: "Alegria", 4: "Calma", 5: "Tristeza", 6: "Sorpresa"}


@app.route('/detect-emotion', methods=['POST'])
def detect_emotions():

    # Parametros Body
    percentage = float(request.form.get('percentage'))
    consecutive_recognition = int(
        request.form.get('consecutiveRecognitionSuccess'))
    emotion_prediction_str = str(request.form.get('emotionPrediction')).lower()
    images = request.files.getlist('images')

    if not images:
        return jsonify({'error': 'No se han enviado imágenes'}), 405

    # Procesamiento de la IA por CNN
    results = {}
    for index, image_file in enumerate(images):
        img = Image.open(image_file)
        gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            'haarcascades/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            gray_img, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            results[f'emotion_{index + 1}'] = 'No se detectó ninguna cara en la imagen.'
            # Guarda la imagen sin cara en el servidor
            # no_face_image_path = os.path.join(
            #     app.config['UPLOAD_FOLDER'], f'no_face_image_{index + 1}.jpg')
            # img.save(no_face_image_path)
        else:
            emotions = []

            for (x, y, w, h) in faces:
                roi_gray = gray_img[y:y+h, x:x+w]
                cropped_img = np.expand_dims(np.expand_dims(
                    cv2.resize(roi_gray, (48, 48)), -1), 0)
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                emotions.append(emotion_dict[maxindex])

            # Encuentra la emoción más común en la imagen
            most_common_emotion = max(set(emotions), key=emotions.count)
            results[f'emotion_{index + 1}'] = most_common_emotion

            # Guardo la imagen procesada en la PC/ Servidor
            # processed_image_path = os.path.join(
            #     app.config['UPLOAD_FOLDER'], f'processed_image_{index + 1}.jpg')
            # img.save(processed_image_path)

    # Analisis de factibilidad
    numberOfHits = 0
    reliability = 0.0
    consecutiveEmotions = 0  # Contador de emociones consecutivas

    # Primer criterio: Obtener una fiabilidad del x%
    for key, emotion in results.items():
        if emotion_prediction_str == str(emotion.lower()):
            numberOfHits += 1
    reliability = (numberOfHits / len(results)) * 100

    # Segundo criterio: Obtener una x cantidad de aciertos consecutivos
    for key, emotion in results.items():
        if emotion_prediction_str == str(emotion.lower()):
            consecutiveEmotions += 1
            if consecutiveEmotions >= consecutive_recognition:
                break
        else:
            consecutiveEmotions = 0  # Reinicia el contador si no se cumple la condición

    response_data = {
        'success': reliability > percentage or consecutiveEmotions >= consecutive_recognition,
        'reliability': round(reliability, 2),
        'consecutive_recognition': consecutiveEmotions,
        'emotion_prediction': request.form.get('emotionPrediction'),
        'results': results
    }
    print(response_data)
    return jsonify(response_data), 200


if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 3001), app)
    http_server.serve_forever()