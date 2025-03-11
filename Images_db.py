import torch
import torchvision.transforms as transforms
from torchvision import models
import timm
from PIL import Image
import json
from config import get_db_connection

# Преобразования для нормализации изображений, чтобы они соответствовали стандарту модели
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Модели для извлечения признаков
def get_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
    elif model_name == 'convnext_base':
        model = timm.create_model('convnext_base', pretrained=True)
    elif model_name == 'regnet_y_16gf':
        model = models.regnet_y_16gf(pretrained=True)
    else:
        raise ValueError(f"Модель {model_name} не поддерживается.")

    model.eval()  # Переводим модель в режим инференса
    return model


# Функция для извлечения признаков изображения
def extract_image_features(image_path, model_name):
    image_path = 'static/' + image_path
    model = get_model(model_name)
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Добавляем размерность для батча
    with torch.no_grad():
        features = model(image)  # Извлекаем признаки
    return features.squeeze().numpy()  # Преобразуем в numpy массив


# Создание таблиц для сохранения векторов
def create_name_and_uniq_name_tables(name):
    conn = get_db_connection()
    cur = conn.cursor()

    create_name_table = f"""
    CREATE TABLE IF NOT EXISTS {name} (
        id SERIAL PRIMARY KEY,
        image_path VARCHAR,
        vector JSONB
    );
    """

    try:
        cur.execute(create_name_table)
        conn.commit()
        print(f"Таблица '{name}' успешно создана.")
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при создании таблицы '{name}': {e}")
    finally:
        cur.close()


# Сохранение векторов в таблицу
def process_vectors(name, model_name, limit=1000):
    conn = get_db_connection()
    cur = conn.cursor()
    image_count = 0

    # Получаем уникальные image_path из таблицы uniq_sentence_t5_base
    cur.execute("SELECT image_path FROM uniq_sentence_t5_base ORDER BY image_path ASC LIMIT %s;", (limit,))
    image_paths = cur.fetchall()

    for (image_path,) in image_paths:
        if image_count >= limit:
            break

        # Пропускаем, если изображение уже обработано
        cur.execute(f"SELECT 1 FROM {name} WHERE image_path = %s;", (image_path,))
        if cur.fetchone():
            continue

        # Извлекаем признаки для текущего изображения
        features = extract_image_features(image_path, model_name)
        vector_json = json.dumps(features.tolist())  # Преобразуем вектор в формат JSON
        # Сохраняем векторы в таблицу
        cur.execute(f"""
            INSERT INTO {name} (image_path, vector)
            VALUES (%s, %s)
        """, (image_path, vector_json))

        image_count += 1
        print(image_count)

    conn.commit()
    cur.close()
    conn.close()
    print(f"Векторы сохранены в таблицу {name} для модели {model_name}.")


# Основная часть программы
if __name__ == '__main__':
    models_list = [
        'regnet_y_16gf',
        'resnet50',
        'vgg16',
        'efficientnet_b0',
        'inception_v3',
        'convnext_base'
    ]

    for model_name in models_list:
        name = f"{model_name}_vectors"  # Имя таблицы для модели

        # Создание таблицы для текущей модели
        create_name_and_uniq_name_tables(name)

        # Процесс векторизации и сохранения в таблицу для текущей модели
        process_vectors(name, model_name)

        print(f"Все операции для модели {model_name} выполнены успешно!")

