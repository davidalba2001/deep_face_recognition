import cv2

# Cargar el modelo entrenado
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('output/modelo_entrenado.yml')

# Cargar el detector de rostros de Viola-Jones
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar el mapeo de nombres y etiquetas
label_to_name = {}
with open('output/nombres.txt', 'r') as f:
    for line in f:
        label, name = line.strip().split(':')
        label_to_name[int(label)] = name

# Iniciar la cámara
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detección de rostros
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y+h, x:x+w]
        
        # Reconocer rostro
        label, confidence = recognizer.predict(face_roi)
        
        # Si la confianza es menor a 100, mostrar el nombre
        if confidence < 100:
            name = label_to_name.get(label, "Desconocido")
            label_text = f'{name}, Confianza: {round(100 - confidence, 2)}%'
        else:
            label_text = 'Desconocido'
        
        # Dibujar recuadro y texto
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    
    # Mostrar imagen
    cv2.imshow('Reconocimiento Facial', frame)
    
    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar cámara y ventanas
cam.release()
cv2.destroyAllWindows()

