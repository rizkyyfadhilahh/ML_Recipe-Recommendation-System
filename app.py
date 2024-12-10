# Mengimpor library yang dibutuhkan
from flask import Flask, render_template, request  # Flask untuk membuat aplikasi web
from sklearn.neighbors import NearestNeighbors  # Algoritma KNN untuk mencari resep terdekat
from sklearn.preprocessing import StandardScaler  # Untuk normalisasi fitur numerik
from sklearn.feature_extraction.text import TfidfVectorizer  # Untuk mengubah daftar bahan menjadi fitur vektor
import numpy as np  # Untuk operasi numerik
import pandas as pd  # Untuk manipulasi data (pandas digunakan untuk data frame)

# Membuat instance aplikasi Flask
app = Flask(__name__) 
app.config['DEBUG'] = True  # Mengaktifkan mode debug Flask

# Memuat dataset resep dari file CSV
data = pd.read_csv("recipe_final (1).csv")

# Preprocess bahan resep
vectorizer = TfidfVectorizer()  # Menggunakan TfidfVectorizer untuk mengubah daftar bahan menjadi vektor numerik
X_ingredients = vectorizer.fit_transform(data['ingredients_list'])  # Mengubah kolom 'ingredients_list' menjadi vektor

# Normalisasi fitur numerik
scaler = StandardScaler()  # Standarisasi fitur numerik seperti kalori, lemak, dll.
X_numerical = scaler.fit_transform(data[['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']])  # Normalisasi kolom-kolom tersebut

# Menggabungkan fitur numerik dan fitur bahan menjadi satu array
X_combined = np.hstack([X_numerical, X_ingredients.toarray()])  # Menggabungkan fitur numerik dan fitur bahan (TF-IDF) dalam satu array

# Melatih model KNN (K-Nearest Neighbors) untuk mencari resep yang mirip
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')  # KNN dengan 3 tetangga terdekat dan menggunakan jarak Euclidean
knn.fit(X_combined)  # Melatih model dengan data gabungan

# Fungsi untuk merekomendasikan resep berdasarkan input dan alergi
def recommend_recipes(input_features, allergies): 
    input_features_scaled = scaler.transform([input_features[:7]])  # Normalisasi fitur numerik input (kalori, lemak, dll.)
    input_ingredients_transformed = vectorizer.transform([input_features[7]])  # Mengubah bahan input menjadi fitur TF-IDF
    input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])  # Menggabungkan fitur numerik dan bahan
    distances, indices = knn.kneighbors(input_combined)  # Mencari 3 resep terdekat berdasarkan input

    # Ambil resep yang direkomendasikan berdasarkan indeks yang ditemukan
    recommendations = data.iloc[indices[0]]  # Mengambil data resep yang sesuai dengan indeks yang ditemukan

    # Filter resep berdasarkan alergi yang dipilih
    if allergies:  # Jika ada alergi yang dipilih
        # Menghapus resep yang mengandung bahan yang sesuai dengan alergi yang dipilih
        recommendations = recommendations[~recommendations['ingredients_list'].str.contains('|'.join(allergies), case=False)]
    
    return recommendations[['recipe_name', 'ingredients_list', 'image_url']].head(5)  # Mengembalikan 5 rekomendasi resep teratas

# Fungsi untuk memotong teks jika panjangnya melebihi batas tertentu
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."  # Jika panjang teks lebih dari batas, maka dipotong dan ditambahkan "..."
    else:
        return text  # Jika panjang teks sudah sesuai, kembalikan teks aslinya

# Mendefinisikan route untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':  # Jika request menggunakan metode POST (form dikirim)
        # Mengambil data dari form input (fitur numerik)
        calories = float(request.form['calories'])
        fat = float(request.form['fat'])
        carbohydrates = float(request.form['carbohydrates'])
        protein = float(request.form['protein'])
        cholesterol = float(request.form['cholesterol'])
        sodium = float(request.form['sodium'])
        fiber = float(request.form['fiber'])
        ingredients = request.form['ingredients']  # Mengambil daftar bahan dari input
        allergies = request.form.getlist('allergies[]')  # Mengambil alergi yang dipilih oleh pengguna
        
        # Menyiapkan fitur input untuk rekomendasi
        input_features = [calories, fat, carbohydrates, protein, cholesterol, sodium, fiber, ingredients] 
        recommendations = recommend_recipes(input_features, allergies)  # Mendapatkan rekomendasi resep berdasarkan input dan alergi
        
        # Menampilkan hasil rekomendasi ke halaman web
        return render_template('index.html', recommendations=recommendations.to_dict(orient='records'), truncate=truncate) 
    
    return render_template('index.html', recommendations=[])  # Jika request menggunakan metode GET, tampilkan halaman awal tanpa rekomendasi

# Menjalankan aplikasi Flask
if __name__ == '__main__': 
    app.run(debug=True)  # Menjalankan server Flask dengan mode debug yang aktif
