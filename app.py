import uuid
from flask import Flask, request, jsonify
from google.cloud import storage, firestore
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from io import StringIO
import tempfile
import logging

app = Flask(__name__)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'serviceaccountkey.json'
BUCKET_NAME = 'storage-ml-similliar'
CSV_FILE_PATH = 'dataset-book/dataset_book.csv'

logging.basicConfig(level=logging.INFO)

def initialize_firestore():
    try:
        db = firestore.Client()
        logging.info("Firestore initialized successfully.")
        return db
    except Exception as e:
        logging.error(f"Error initializing Firestore: {e}")
        return None

db = initialize_firestore()
if db is None:
    logging.error("Firestore initialization failed. Exiting.")
    exit(1)

try:
    model = tf.keras.models.load_model('book_recommendation_model.h5')
    logging.info("ML model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading ML model: {e}")
    exit(1)

def download_blob(bucket_name, source_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    return blob.download_as_text()

def upload_csv(bucket_name, destination_blob_name, data_frame):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    csv_buffer = StringIO()
    data_frame.to_csv(csv_buffer, index=False)
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logging.info(f"File {source_file_name} uploaded to {destination_blob_name}.")

try:
    csv_data = download_blob(BUCKET_NAME, CSV_FILE_PATH)
    df = pd.read_csv(StringIO(csv_data))
    logging.info("CSV data loaded successfully.")
except Exception as e:
    logging.error(f"Error loading CSV data: {e}")
    df = pd.DataFrame()

@app.route("/")
def index():
    return jsonify({"status": {"code": 200, "message": "API is running"}}), 200

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or not request.form:
        return jsonify({'error': 'Invalid request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)
    upload_blob(BUCKET_NAME, temp_path, file.filename)

    valid_genres = [
        'fiction',
        'non-fiction',
        'children\'s',
        'education',
        'religion',
        'comics',
        'art',
        'health',
        'technology',
        'other'
    ]

    try:
        genre = request.form['genre']
        if genre not in valid_genres:
            return jsonify({'error': f'Invalid genre. Valid genres are: {", ".join(valid_genres)}'}), 400

        book_info = {
            'name': request.form['name'],
            'id': str(uuid.uuid4()),
            'author': request.form['author'],
            'rating': float(request.form['rating']),
            'user': request.form['user'],
            'genre': genre  
        }
        logging.info(f"Book Info: {book_info}")

        db.collection('books').document(book_info['id']).set(book_info)

        global df
        new_row = {
            'user': book_info['user'],
            'name': book_info['name'],
            'review/score': book_info['rating'],
            'Genre': book_info['genre']  
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        upload_csv(BUCKET_NAME, CSV_FILE_PATH, df)
    except Exception as e:
        logging.error(f"Error in /upload: {e}")
        return jsonify({'error': f'Failed to process data: {e}'}), 500
    finally:
        os.remove(temp_path)

    return jsonify({'message': 'File uploaded successfully', 'book_info': book_info}), 200


@app.route('/get_buku', methods=['GET'])
def get_buku():
    try:
        books = db.collection('books').stream()
        book_list = [book.to_dict() for book in books]
        logging.info(f"Books fetched: {book_list}")
        return jsonify({'message': 'Daftar buku', 'data': book_list}), 200
    except Exception as e:
        logging.error(f"Error in /get_buku: {e}")
        return jsonify({'error': f'Failed to fetch books: {e}'}), 500

@app.route('/rekomendasi', methods=['GET'])
def rekomendasi():
    try:
        book_title = request.args.get('book_title')
        if not book_title:
            return jsonify({'error': 'Parameter "book_title" is required'}), 400

        logging.info(f"Request received for book_title: {book_title}")

        if 'Title' not in df.columns or 'Genre' not in df.columns:
            return jsonify({'error': 'Required columns not found in dataset'}), 500

        book_features = df[df['Title'] == book_title]
        if book_features.empty:
            return jsonify({'error': 'Book not found in dataset'}), 404

        book_genre = book_features['Genre'].values[0]

        similar_genre_books = df[df['Genre'] == book_genre]

        features_to_predict = similar_genre_books.iloc[:, 1:].values 

        predictions = model.predict(features_to_predict)

        recommended_indices = np.argsort(predictions.flatten())[-6:-1]  
        recommended_books = similar_genre_books.iloc[recommended_indices]

        return jsonify(recommended_books.to_dict(orient='records')), 200

    except Exception as e:
        logging.error(f"Error in /rekomendasi: {e}")
        return jsonify({'error': f'Failed to get recommendations: {e}'}), 500



@app.route('/rating', methods=['POST'])
def rating():
    try:
        data = request.json
        book_name = data.get('name', None)
        if not book_name:
            return jsonify({'error': 'Missing key: name'}), 400

        if 'name' not in df.columns or 'review/score' not in df.columns:
            return jsonify({'error': 'Required columns not found in dataset'}), 500

        review_scores = df[df['name'] == book_name]['review/score']
        if review_scores.empty:
            return jsonify({'error': 'No reviews found for this book'}), 404

        average_rating = review_scores.mean()
        min_rating = df['review/score'].min()
        max_rating = df['review/score'].max()
        normalized_score = (average_rating - min_rating) / (max_rating - min_rating) if max_rating != min_rating else 0

        return jsonify({'average_rating': average_rating, 'normalized_score': normalized_score}), 200
    except Exception as e:
        logging.error(f"Error in /rating: {e}")
        return jsonify({'error': f'Failed to calculate rating: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
