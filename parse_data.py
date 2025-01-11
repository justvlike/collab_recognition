import librosa
import numpy as np
import glob
import os

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


def load_audio_and_convert_to_spectrogram(audio_file, n_mels=128):
    # Загружаем аудио через librosa
    y, sr = librosa.load(audio_file, sr=None)  # Load with original sampling rate
    # Создаем Mel-spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    # Переводим в log scale для лучшего представления
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram

def prepare_training_data(training_path, n_mels=128):
    X = []  # Features (spectrograms)
    y = []  # Labels (instrument classes)

    # Маппим в соответствии с папками в архиве
    instrument_map = {
        'cel': 0, 'cla': 1, 'flu': 2, 'gac': 3, 'gel': 4,
        'org': 5, 'pia': 6, 'sax': 7, 'tru': 8, 'vio': 9, 'voi': 10
    }

    # Проход по всем подкаталогам и непосредственно преобразование
    for instrument, label in instrument_map.items():
        folder_path = os.path.join(training_path, instrument)
        for audio_file in glob.glob(os.path.join(folder_path, '*.wav')):
            spectrogram = load_audio_and_convert_to_spectrogram(audio_file, n_mels=n_mels)
            X.append(spectrogram)
            y.append(label)

    return np.array(X), np.array(y)

# Подготовка данных, вызов функции преобразования
dataset_path = "E:\\Files\\PycharmProjects\\collab_recognition\\files"
training_path = os.path.join(dataset_path, 'IRMAS-TrainingData')
X, y = prepare_training_data(training_path)
print(f"Data prepared: {X.shape}, Labels: {y.shape}")

# Нормализация спектрограмм к облатстям [0, 1]
X = X / np.max(X)

# Разделим данные (80% на обучение, 20% на проверку)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Добавляем параметр для дальнейшего использования сверточной НС
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

# Модель сверточной НС
def build_model(input_shape, num_classes):
    model = models.Sequential([
        # Первый сверточный блок
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Второй сверточный блок
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Третий сверточный блок
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Дополнительные слои
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.0001)  # Уменьшите learning rate для стабильности
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Преобразование y_train и y_val в более удобный формат.
y_train = to_categorical(y_train, num_classes=len(set(y)))
y_val = to_categorical(y_val, num_classes=len(set(y)))

# Строим модель
input_shape = X_train.shape[1:]  # Размер спектрограмм
num_classes = len(set(y))  # Количество типов классов
model = build_model(input_shape, num_classes)
model.summary()

# Обучаем модель
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,  # Больше проходов - сильнее обучение
    batch_size=32
)
X_train_small = X_train[:500]
y_train_small = y_train[:500]
model.fit(X_train_small, y_train_small, validation_data=(X_val, y_val), epochs=5, batch_size=32)


# График точности
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# График валидации
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Точность
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Сохраняем модель
model.save(os.path.join(dataset_path, 'instrument_classifier_model.h5'))

# Снова указываем все доступные варианты
instrument_map = {
    'cel': 0, 'cla': 1, 'flu': 2, 'gac': 3, 'gel': 4,
    'org': 5, 'pia': 6, 'sax': 7, 'tru': 8, 'vio': 9, 'voi': 10
}

# Инвертируем эту map
instrument_map_inv = {v: k for k, v in instrument_map.items()}

# Загружаем тестовый файл
test_audio_file = 'E:\\Files\\PycharmProjects\\collab_recognition\\files\\IRMAS-TrainingData\\gel\\001__[gel][dru][pop_roc]0829__1.wav'
test_spectrogram = load_audio_and_convert_to_spectrogram(test_audio_file)
test_spectrogram = test_spectrogram / np.max(test_spectrogram)  # Нормализуем
test_spectrogram = test_spectrogram[np.newaxis, ..., np.newaxis]

# Делаем предположение
predicted_label = model.predict(test_spectrogram).argmax()
print(f"Predicted Instrument: {instrument_map_inv[predicted_label]}")

print(f"X_train min: {X_train.min()}, max: {X_train.max()}, shape: {X_train.shape}")
print(f"X_val min: {X_val.min()}, max: {X_val.max()}, shape: {X_val.shape}")

print(f"y_train shape: {y_train.shape}, unique values: {np.unique(y_train)}")
print(f"y_val shape: {y_val.shape}, unique values: {np.unique(y_val)}")
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


