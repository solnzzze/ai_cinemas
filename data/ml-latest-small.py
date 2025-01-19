import requests
import zipfile
import os

url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
output_file = "ml-latest-small.zip"

print("Скачивание файла...")
response = requests.get(url)
with open(output_file, "wb") as f:
    f.write(response.content)
print(f"Файл {output_file} успешно скачан.")

# Путь к ZIP-архиву
zip_path = os.path.join('data', 'ml-latest-small.zip')

# Путь для извлечения
extract_path = os.path.join('data', 'ml-latest-small')

# Извлечение архива
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
