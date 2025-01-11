import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score


# Функция для загрузки аудио и преобразования в мел-спектрограмму
def load_audio_and_convert_to_spectrogram(audio_file, n_mels=128):
    try:
        y, sr = librosa.load(audio_file, sr=22050)
        if len(y) == 0 or np.all(y == 0):
            print(f"Audio file {audio_file} is empty or contains only silence.")
            return None

        # Нормализация аудиосигнала
        y = librosa.util.normalize(y)

        # Создание спектрограммы
        spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512
        )
        log_spectrogram = librosa.power_to_db(spectrogram + 1e-6, ref=np.max)

        # Проверка амплитуды спектрограммы
        if np.max(log_spectrogram) < -80:  # Порог минимальной амплитуды
            print(f"Warning: Spectrogram of {audio_file} has too low amplitude.")
            return None

        return log_spectrogram
    except Exception as e:
        print(f"Error loading {audio_file}: {e}")
        return None

# Словарь классов инструментов
instrument_map = {
    'cel': 0, 'cla': 1, 'flu': 2, 'gac': 3, 'gel': 4,
    'org': 5, 'pia': 6, 'sax': 7, 'tru': 8, 'vio': 9, 'voi': 10
}
instrument_map_inv = {v: k for k, v in instrument_map.items()}  # Обратный словарь

# Загрузка модели
model_path = "E:\\Files\\PycharmProjects\\collab_recognition\\files\\instrument_classifier_model.h5"
model = load_model(model_path)

# Подготовка тестовых данных
test_data_path = "E:\\Files\\PycharmProjects\\collab_recognition\\files\\IRMAS-TrainingData"
true_labels = []
predicted_labels = []
files_limit = 50  # Лимит на количество файлов

# Прогон по всем подкаталогам в тестовой папке
for instrument, label in instrument_map.items():
    folder_path = os.path.join(test_data_path, instrument)
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

    # Ограничиваем количество файлов до 50
    for audio_file in audio_files[:files_limit]:
        file_path = os.path.join(folder_path, audio_file)

        # Преобразование в спектрограмму
        test_spectrogram = load_audio_and_convert_to_spectrogram(file_path)
        if test_spectrogram is None:
            continue  # Пропускаем проблемный файл

        # Нормализация
        if np.max(test_spectrogram) > 1e-6:
            test_spectrogram = test_spectrogram / np.max(test_spectrogram)
        else:
            print(f"Warning: Spectrogram of {file_path} has too low amplitude.")
            continue

        test_spectrogram = test_spectrogram[np.newaxis, ..., np.newaxis]  # Добавление осей

        # Предсказание
        predicted_label = model.predict(test_spectrogram).argmax()
        true_labels.append(label)
        predicted_labels.append(predicted_label)

# Вычисление точности
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Testing Accuracy: {accuracy * 100:.2f}%")
