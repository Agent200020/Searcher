import os

UPLOAD_FOLDER = 'uploads'

MAX_FOLDER_SIZE = 10 * 1024 * 1024

# Функция для проверки размера папки
def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size

# Функция для удаления старых файлов, если размер папки превышает лимит
def clean_upload_folder():
    folder_size = get_folder_size(UPLOAD_FOLDER)

    if folder_size > MAX_FOLDER_SIZE:
        print(f"Размер папки {UPLOAD_FOLDER} превышает лимит. Размер: {folder_size} байт")

        # Получаем список файлов с их временем последней модификации
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                files.append((file_path, os.path.getmtime(file_path)))

        # Сортируем файлы по времени последней модификации (по возрастанию)
        files.sort(key=lambda x: x[1])

        # Удаляем старые файлы, пока размер папки не станет меньше лимита
        for file_path, _ in files:
            if folder_size <= MAX_FOLDER_SIZE:
                break
            os.remove(file_path)
            print(f"Удалён файл {file_path}")
            folder_size = get_folder_size(UPLOAD_FOLDER)

