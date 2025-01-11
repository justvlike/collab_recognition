import zipfile
zip_file_path = "E:\\Files\\PycharmProjects\\collab_recognition\\files\\IRMAS-TrainingData.zip"
extract_folder = "E:\\Files\\PycharmProjects\\collab_recognition\\files"
# Разархивация загруженного архива
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)
