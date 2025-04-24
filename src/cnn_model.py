from keras.utils import image_dataset_from_directory, to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Obteber el directorio actual
current_dir = Path(__file__).parent
models_dir = current_dir.parent / 'trained_model_parameters'
output_path = Path(current_dir.parent / 'trained_model_parameters/best_model.h5')

dt = datetime.now()
ts = datetime.timestamp(dt)


def train_model():
    """
    Train a CNN model for gesture recognition using images from the 'data' directory.
    The model is saved as 'hand_sign_model.h5' after training.
    """
    # Create an image data generator for loading and augmenting images

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train = datagen.flow_from_directory("data", target_size=(640, 480), batch_size=32, class_mode='categorical', subset='training')
    val = datagen.flow_from_directory("data", target_size=(640, 480), batch_size=32, class_mode='categorical', subset='validation')

    filtros = 32
    regularizers_w  = 1e-4
    n_clases = train.num_classes

    # Definición del modelo
    model = Sequential()
    
    # Capa convolucional 1
    model.add(Conv2D(
        filters = filtros, 
        kernel_size = (3, 3),
        strides = (3, 3), 
        padding = 'same', 
        kernel_regularizer = regularizers.l2(regularizers_w), 
        input_shape = (640, 480,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # Capa convolucional 2
    model.add(Conv2D(
        filters = filtros, 
        kernel_size = (3, 3),
        strides = (3, 3), 
        padding = 'same', 
        kernel_regularizer = regularizers.l2(regularizers_w)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    # Capa convolucional 3
    model.add(Conv2D(
        filters = filtros*2, 
        kernel_size = (3, 3),
        strides = (3, 3), 
        padding = 'same', 
        kernel_regularizer = regularizers.l2(regularizers_w)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Capa convolucional 4
    model.add(Conv2D(
        filters = filtros*2, 
        kernel_size = (3, 3),
        strides = (3, 3), 
        padding = 'same', 
        kernel_regularizer = regularizers.l2(regularizers_w)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    # Capa convolucional 5
    model.add(Conv2D(
        filters = filtros*4, 
        kernel_size = (3, 3),
        strides = (3, 3), 
        padding = 'same', 
        kernel_regularizer = regularizers.l2(regularizers_w)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # Capa convolucional 6
    model.add(Conv2D(
        filters = filtros*4, 
        kernel_size = (3, 3),
        strides = (3, 3), 
        padding = 'same', 
        kernel_regularizer = regularizers.l2(regularizers_w)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    # Capa de flatten
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_clases, activation='softmax'))

    # Resumen del modelo
    model.summary()

   # Compilación del modelo
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(),
        metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint(
        filepath=str(output_path),
        monitor='accuracy',
        save_best_only=True,
        verbose=1)

    # Entrenamiento del modelo
    history = model.fit(
        train,
        steps_per_epoch=len(train),
        batch_size=32,
        epochs=100,
        validation_data=val,
        verbose = 2,
        shuffle=True,
        callbacks=[model_checkpoint])
 
    # Guardar el modelo
    model_path = models_dir / f'gesture_recognition.keras'
    model.save(model_path)

    # Gráfica de la precisión del modelo
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy')
    plt.show()
