import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Importe
preprocess_input
from tensorflow.keras.utils import to_categorical

# Tamanho das imagens de entrada
image_size = (418, 200)
batch_size = 32
num_classes = 20  # 0 a 16 bois

import cv2
import os

# Diretório que contém suas imagens
data_dir = 'DataSet'

images = []
labels = [20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,0,20,20,0,20,20,20,0,20,0,20,20,0,20,0,20,20,20,20,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,5,5,5,5,5,5,5,5,5,5,5,0,5,0,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,0,5,5,20,20,20,20,20,20,20,20,20,20,20,20,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,0,0,0,20,20,20,20,20,20,0]

# Número total de imagens
num_images = len(labels)

for i in range(num_images):
    image_path = os.path.join(data_dir, f'{i}.jpg')
    image = cv2.imread(image_path)
    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image_rgb)
        # Suponhamos que você tenha as etiquetas em uma lista correspondente
        #labels.append(labels)


# Divisão de dados em conjuntos de treinamento, validação e teste
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

X_train = np.array([preprocess_input(img) for img in X_train])
X_val = np.array([preprocess_input(img) for img in X_val])
X_test = np.array([preprocess_input(img) for img in X_test])

# Converta as etiquetas em arrays NumPy
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)


# Carregamento de um modelo pré-treinado (MobileNetV2) e substituição da camada de saída
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Congela as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Compilação do modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
epochs = 100  # Ajuste conforme necessário
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)

# Avaliação do modelo usando o conjunto de teste
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Acurácia no conjunto de teste:", test_accuracy)

# Salva o modelo treinado
model.save('money_md.h5')
