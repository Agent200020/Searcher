import numpy as np
from numpy.linalg import norm
import faiss
import hnswlib
from sklearn.neighbors import KDTree

def cosine_similarity1(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-10)


def cosine_filling(query_vector, rows, results):
    for row in rows:
        id_, image_path, vector_json = row

        sim = cosine_similarity1(query_vector, vector_json)
        results.append({
            "id": id_,
            "image_path": image_path,
            "similarity": sim
        })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)


def manhattan_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return 1/(1+np.sum(np.abs(vec1 - vec2)))


def manhattan_filling(query_vector, rows, results):
    for row in rows:
        id_, image_path, vector_json = row
        sim = manhattan_distance(query_vector, vector_json)
        results.append({
            "id": id_,
            "image_path": image_path,
            "similarity": sim
        })
    return sorted(results, key=lambda x: x["similarity"], reverse=True)


def euclidean_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return 1/(1+np.sqrt(np.sum((vec1 - vec2) ** 2)))


def euclidean_filling(query_vector, rows, results):
    for row in rows:
        id_, image_path, vector_json = row
        sim = euclidean_distance(query_vector, vector_json)
        results.append({
            "id": id_,
            "image_path": image_path,
            "similarity": sim
        })
    return sorted(results, key=lambda x: x["similarity"], reverse=True)


def kd_tree_search(query_vector, dataset, rows, results, k=10):
    tree = KDTree(dataset)
    distances, indices = tree.query([query_vector], k=k)
    distances, indices = distances[0], indices[0]

    for i, (idx, dist) in enumerate(zip(indices, distances)):
        id_, image_path, _ = rows[idx]  # Получаем данные, _ - пропуск, т.к. три элемента в строке

        sim = 1 / (1 + dist)  # Пример: преобразуем расстояние в сходство, если это необходимо
        results.append({
            "id": id_,
            "image_path": image_path,
            "similarity": sim
        })
    return results


def hnsw_search(query_vector, dataset, rows, results, k=10):

    dim = len(dataset[0])  # Размерность векторов
    index = hnswlib.Index(space='l2', dim=dim)  # Используем L2 расстояние (евклидово)
    index.init_index(max_elements=len(dataset), ef_construction=200, M=16)  # Настройка индекса
    index.add_items(np.array(dataset))  # Добавляем векторы

    labels, distances = index.knn_query(np.array([query_vector]), k=k)

    hnsw_results = [(labels[0][i], 1 / (1 + distances[0][i])) for i in range(len(labels[0]))]

    for idx, sim in hnsw_results:
        id_, image_path, _ = rows[idx]  # Получаем данные, _ - пропуск, т.к. три элемента в строке

        results.append({

            "id": id_,
            "image_path": image_path,
            "similarity": float(sim)  # Преобразование к типу float

        })

    return results


def faiss_search(query_vector, dataset, rows, results, k=10):

    index = faiss.IndexFlatL2(len(query_vector))  # Создаём индекс для L2 (евклидовых) расстояний
    dataset_float32 = np.array(dataset).astype('float32')  # Преобразуем все векторы в float32
    index.add(dataset_float32)  # Добавляем данные в индекс

    distances, labels = index.search(np.array([query_vector]).astype('float32'), k)
    faiss_results = [(labels[0][i], 1 / (1 + distances[0][i])) for i in range(len(labels[0]))]

    for idx, sim in faiss_results:
        id_, image_path, _ = rows[idx]  # Получаем данные, _ - пропуск, т.к. три элемента в строке
        results.append({
            "id": id_,
            "image_path": image_path,
            "similarity": float(sim)
        })
    return results

''' Сложность алгоритмов поиска
Косинуное O(m*n^2)
Манхэттэнское O(m*n^2)
Евклидово O(m*n^2)
m - кол-во строк, n - размерность векторов

KD-дерево O(m * log(m) + log(m) * k)
HNSW O(m * log(m) + log(m) * k)
FAISS O(m + log(m) * k)
m - количество элементов, k - количество ближайших соседей

FAISS наиболее эффективный

'''


