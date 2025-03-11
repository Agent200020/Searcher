import json
import numpy as np
from sentence_transformers import SentenceTransformer
from config import get_db_connection


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

    create_uniq_name_table = f"""
    CREATE TABLE IF NOT EXISTS uniq_{name} (
        id SERIAL PRIMARY KEY,
        image_path VARCHAR,
        vector JSONB
    );
    """

    try:
        cur.execute(create_name_table)
        cur.execute(create_uniq_name_table)
        conn.commit()
        print(f"Таблицы '{name}' и 'uniq_{name}' успешно созданы.")
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при создании таблиц '{name}' и 'uniq_{name}': {e}")
    finally:
        cur.close()



def process_vectors(name, model_name):
    conn = get_db_connection()
    cur = conn.cursor()

    model = SentenceTransformer(model_name)

    cur.execute("SELECT id, image_path, description FROM description ORDER BY id ASC LIMIT 5000;")
    rows = cur.fetchall()

    for id_, image_path, description in rows:
        description_vector = model.encode(description).tolist()
        vector_json = json.dumps(description_vector)

        cur.execute(f"""
            INSERT INTO {name} (image_path, vector)
            VALUES (%s, %s)
        """, (image_path, vector_json))

    conn.commit()
    cur.close()
    conn.close()
    print(f"Векторы сохранены в таблицу {name}.")

def create_unique_vectors(name):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(f"SELECT DISTINCT image_path FROM {name} ORDER BY image_path ASC LIMIT 1000;")
    image_paths = cur.fetchall()

    for (image_path,) in image_paths:
        cur.execute(f"SELECT vector FROM {name} WHERE image_path = %s;", (image_path,))
        vectors = [row[0] for row in cur.fetchall()]

        mean_vector = np.mean(vectors, axis=0).tolist()
        vector_json = json.dumps(mean_vector)

        cur.execute(f"""
            INSERT INTO uniq_{name} (image_path, vector)
            VALUES (%s, %s)
        """, (image_path, vector_json))

    conn.commit()
    cur.close()
    conn.close()
    print(f"Уникальные усреднённые векторы сохранены в таблицу uniq_{name}.")


def preprocess():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id SERIAL PRIMARY KEY,
                image_path TEXT NOT NULL,
                description TEXT NOT NULL
            );
        """)

    with open('captions.txt', 'r') as file:
        for line in file:
            image_filename, caption = line.strip().split(",", 1)
            image_path = f"Images/{image_filename}"
            cur.execute("""
                INSERT INTO images (image_path, description)
                VALUES (%s, %s)
            """, (image_path, caption))

    conn.commit()
    cur.close()
    conn.close()


model_names = [
    ('paraphrase-MiniLM-L6-v2', 'paraphrase_MiniLM_L6_v2'),
    ('all-MiniLM-L6-v2', 'all_MiniLM_L6_v2'),
    ('paraphrase-mpnet-base-v2', 'paraphrase_mpnet_base_v2'),
    ('all-mpnet-base-v2', 'all_mpnet_base_v2'),
    ('sentence-t5-base', 'sentence_t5_base'),
    ('sentence-t5-large', 'sentence_t5_large')
]

if __name__ == '__main__':
    preprocess()
    for model_name, name in model_names:
        create_name_and_uniq_name_tables(name)
        process_vectors(name, model_name)
        create_unique_vectors(name)
        print(f"Операции для модели {model_name} выполнены успешно!")

    print("Все операции выполнены успешно!")

