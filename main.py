"""
    Main script for the gesture recognition application.
    Usage examples:
    - To collect data: python main.py pulgar_arriba
    - To train model:  python main.py train
    - To predict:      python main.py real_time
"""

from src.dataset_creation import create_image
from src.cnn_model import train_model
from src.real_time_prediction import real_time_prediction
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        print("Argumento proporcionado:", command)

        if command == "train":
            train_model()
        elif command == "real_time":
            real_time_prediction()
        else:
            # Asume que cualquier otro comando es una etiqueta para capturar imágenes
            label = command
            print(f"Capturando imágenes para: {label}")
            create_image(label)
    else:
        print("Uso:")
        print(" - Para capturar datos: python main.py <nombre_gesto>")
        print(" - Para entrenar modelo: python main.py train")
        print(" - Para predecir en tiempo real: python main.py real_time")
