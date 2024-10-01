import cv2
import os
import numpy as np

# Directorios
dataset_dir = 'data/'  # Carpeta donde están las imágenes de entrenamiento
trained_model = 'output/modelo_entrenado.yml'  # Archivo donde se guardará el modelo entrenado
names_file = 'output/nombres.txt'  # Archivo donde se guardará el mapeo de nombres


if not os.path.exists('output'):
    os.makedirs('output')



# Cargar el detector de rostros de Viola-Jones
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Crear el reconocedor LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Mapeo de etiquetas numéricas a nombres de compañeros
label_to_name = {}
current_label = 0

# Función para obtener las imágenes y etiquetas de entrenamiento
def get_images_and_labels(directory):
    face_samples = []
    labels = []
    global current_label

    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if not os.path.isdir(subdir_path):
            continue  # Saltar si no es una carpeta

        label_to_name[current_label] = subdir  # Asignar el nombre al label actual
        
        for file in os.listdir(subdir_path):
            if file.endswith('jpg') or file.endswith('png'):
                img_path = os.path.join(subdir_path, file)
                img = cv2.imread(img_path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detectar rostros en la imagen
                faces = face_cascade.detectMultiScale(gray_img)
                for (x, y, w, h) in faces:
                    face_samples.append(gray_img[y:y+h, x:x+w])  # Extraer la cara
                    labels.append(current_label)

        current_label += 1  # Incrementar el label para el próximo compañero

    return face_samples, labels

# Obtener las imágenes y etiquetas
faces, labels = get_images_and_labels(dataset_dir)

# Entrenar el modelo LBPH
recognizer.train(faces, np.array(labels))

# Guardar el modelo entrenado
recognizer.save(trained_model)

# Guardar el mapeo de nombres
with open(names_file, 'w') as f:
    for label, name in label_to_name.items():
        f.write(f'{label}:{name}\n')

print(f'Modelo entrenado y guardado en {trained_model}')
print(f'Nombres y etiquetas guardados en {names_file}')
