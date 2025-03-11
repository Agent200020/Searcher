from Clean_Upload import *
from methods import *
from Images_methods import *
from googletrans import Translator
from sentence_transformers import SentenceTransformer
from flask import send_from_directory
import json
from flask import Flask, request, jsonify, render_template
import numpy as np
from werkzeug.utils import secure_filename
import os


app = Flask(__name__)

# Устанавливаем папку для загрузки изображений
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('data/flowers', filename)
def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

# Маршрут для загрузки изображения
@app.route('/upload_image', methods=['POST'])
def upload_image():

    image = request.files['file']

    # Сохраняем файл
    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    image_folder = 'uploads/'
    query_image_path = os.path.join(image_folder, filename)

    model_name = request.form.get('model')
    top_results = find_similar_images(query_image_path, model_name)
    return jsonify(top_results)

@app.route('/')
def index():
    # Простая страница с формой поиска
    return render_template('Searcher.html')

@app.route('/upload')
def upload_page():
    return render_template('Images.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    method = data.get('similarity_method')
    model_name = data.get('embedding_model')

    translator = Translator()
    res = translator.translate(query, src='auto', dest='en')
    query = res.text

    model = SentenceTransformer(model_name)

    query_vector = model.encode(query)

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(f"SELECT id, image_path, vector FROM uniq_{model_name.replace("-", "_")} ORDER BY id ASC LIMIT 1000;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    dataset = [np.array(row[2]) if isinstance(row[2], list) else np.array(json.loads(row[2])) for row in rows]

    results = []

    if method == 'cosine':
        results = cosine_filling(query_vector, rows, results)

    elif method == 'manhattan':
        results = manhattan_filling(query_vector, rows, results)

    elif method == 'euclidean':
        results = euclidean_filling(query_vector, rows, results)

    elif method == 'kd_tree':
        results = kd_tree_search(query_vector, dataset, rows, results)

    elif method == 'hnsw':
        results = hnsw_search(query_vector, dataset, rows, results)

    elif method == 'faiss':
        results = faiss_search(query_vector, dataset, rows, results)

    if not results:
        return jsonify({"error": "No search results found"}), 404

    top_results = results[:100]

    return jsonify(top_results)


if __name__ == '__main__':
    clean_upload_folder()
    app.run(debug=True)