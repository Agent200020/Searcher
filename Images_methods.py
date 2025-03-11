from config import *
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from Images_db import get_model
def extract_image_features(image_path, model):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Добавляем размерность для батча
    with torch.no_grad():
        features = model(image)  # Извлекаем признаки
    return features.squeeze().numpy()  # Преобразуем вектор в numpy массив

def find_similar_images(image_path, model_name):
    model = get_model(model_name)
    query_features = extract_image_features(image_path, model)  # Извлекаем признаки изображения для поиска

    # Подключаемся к базе данных
    conn = get_db_connection()
    cur = conn.cursor()

    # Выполняем запрос к базе данных для получения всех векторов и путей изображений
    cur.execute(f"SELECT id, image_path, vector FROM {model_name}_vectors ORDER BY id ASC LIMIT 500;")
    rows = cur.fetchall()
    cur.close()

    # Декодируем векторы из JSON в numpy массивы
    dataset = [np.array(row[2]) for row in rows]
    # Сравниваем изображение с векторами из базы данных
    similarities = []
    for idx, (row, vector) in enumerate(zip(rows, dataset)):
        id_ = row[0]  # id изображения
        image_path = row[1]  # Путь к изображению
        similarity = cosine_similarity([query_features], [vector])[0][0]
        similarities.append({
            "id": id_,
            "image_path": image_path,
            "similarity": similarity
        })

    # Сортируем результаты по схожести в убывающем порядке
    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    # Возвращаем топ-100 наиболее похожих изображений
    top_results = similarities[:100]

    return top_results

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

