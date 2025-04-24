import cv2
import numpy as np
from pathlib import Path
from keras.models import load_model
from keras.utils import image_dataset_from_directory
import time

def real_time_prediction():
    """
    Capture video from the webcam and predict chess pieces in real-time with confidence threshold and delay.
    """

    current_dir = Path(__file__).parent
    models_dir = current_dir.parent / 'trained_model_parameters'
    model_path = models_dir / 'gesture_recognition.keras'
    
    model = load_model(model_path)
    train_ds = image_dataset_from_directory(current_dir.parent / 'data')
    labels = train_ds.class_names
    print("Clases detectadas:", labels)

    cap = cv2.VideoCapture(0)
    last_label = None
    last_change_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (480, 640))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        confidence = np.max(pred)

        if confidence > 0.8:
            current_label = labels[np.argmax(pred)]
        else:
            current_label = "No pieza"

        # Si la etiqueta cambió, actualiza y aplica delay
        if current_label != last_label:
            current_time = time.time()
            if current_time - last_change_time > 1:  # 1 segundo
                last_label = current_label
                last_change_time = current_time
        else:
            current_label = last_label

        cv2.putText(frame, f"Prediccion: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Clasificacion de Piezas en Tiempo Real", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
