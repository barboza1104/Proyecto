import cv2

# Intenta abrir la cámara (0 es la cámara integrada)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
else:
    print("Cámara abierta correctamente")
    # Captura un frame
    ret, frame = cap.read()
    cv2.imshow('Cámara', frame)  # Muestra el frame
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
