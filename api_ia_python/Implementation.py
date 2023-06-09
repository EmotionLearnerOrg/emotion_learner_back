import cv2
import numpy as np
from keras.models import model_from_json
import time



def detect_emotions_in_image(path_image):

    emotion_dict = {0: "Enojado", 1: "Disgustado", 2: "Disgustado", 3: "Feliz", 4: "Neutral", 5: "Triste", 6: "Sorprendido"}

    # load json and create model
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # load weights into new model
    emotion_model.load_weights("model/emotion_model.h5")
    # print("modelo cargado")

    image = cv2.imread(path_image)
    if image is None:
        return("Error al cargar la imagen")
    # image = cv2.resize(image, (48, 48))
    
    
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # print(" detect faces available on camera")
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # print(" take each face available on the camera and Preprocess it");
    results = []
    for (x, y, w, h) in num_faces:
        cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # print(" Predict the emotions")
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_label = emotion_dict[maxindex]
        results.append({
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'emotion': emotion_label
        })

    return results