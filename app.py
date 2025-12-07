# =====================================================
# APP.PY — DENGAN FITUR WARNA DOMINAN (COLOR PALETTE)
# =====================================================

from flask import Flask, render_template, request, url_for
import os
import numpy as np
import pandas as pd
from PIL import Image
import colorsys
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import pairwise_distances

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploaded"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =====================================================
# LOAD HYBRID CNN MODEL + FEATURE EXTRACTOR (dense_19)
# =====================================================
hybrid_model = load_model("CNN_Skripsi.h5", compile=False)

feature_extractor = Model(
    inputs=hybrid_model.inputs,
    outputs=hybrid_model.get_layer("dense_19").output
)

# Label Warna & Hex Color
label_warna = ["Biru", "Coklat", "Hijau", "Merah Kehitaman"]
warna_hex_mapping = {
    "Biru": "#0B707A",
    "Coklat": "#DDBA9E",
    "Hijau": "#2B5533",
    "Merah Kehitaman": "#000529"
}

# =====================================================
# LOAD DATASET CSV
# =====================================================
dataset = pd.read_csv("Dataset_skripsi.csv")

if "Tempat" not in dataset.columns:
    raise ValueError("Dataset_skripsi.csv harus memiliki kolom 'Tempat'")


# =====================================================
# PREPROCESSING Gambar
# =====================================================
def preprocess_image(img_path):
    img = Image.open(img_path).resize((256, 256)).convert("RGB")
    arr = np.array(img) / 255.0
    return arr


# =====================================================
# EKSTRAKSI WARNA DOMINAN (3 WARNA)
# =====================================================
def get_dominant_colors(img_path, k=3):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((150, 150))
    arr = np.array(img).reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(arr)
    colors = kmeans.cluster_centers_.astype(int)

    # convert ke hex
    hex_colors = []
    for c in colors:
        hex_colors.append('#%02x%02x%02x' % (c[0], c[1], c[2]))

    return hex_colors


# =====================================================
# COLOR FEATURE (RGB + HSV)
# =====================================================
def extract_color_feature(img_array):
    R = np.mean(img_array[:, :, 0])
    G = np.mean(img_array[:, :, 1])
    B = np.mean(img_array[:, :, 2])

    hsv = np.zeros_like(img_array)
    for i in range(256):
        for j in range(256):
            hsv[i, j] = colorsys.rgb_to_hsv(
                img_array[i, j, 0],
                img_array[i, j, 1],
                img_array[i, j, 2]
            )
    H = np.mean(hsv[:, :, 0])
    S = np.mean(hsv[:, :, 1])
    V = np.mean(hsv[:, :, 2])

    return np.array([R, G, B, H, S, V])


# =====================================================
# HYBRID EMBEDDING
# =====================================================
def extract_embedding(img_path):
    arr = preprocess_image(img_path)
    rgb_feat = np.array([[np.mean(arr[:, :, 0]), np.mean(arr[:, :, 1]), np.mean(arr[:, :, 2])]])
    img_batch = np.expand_dims(arr, axis=0)

    feat = feature_extractor.predict([img_batch, rgb_feat], verbose=0)
    return feat.flatten()


# =====================================================
# CNN WARNA OUTPUT
# =====================================================
def prediksi_warna(img_path):
    arr = preprocess_image(img_path)
    rgb_feat = np.array([[np.mean(arr[:, :, 0]), np.mean(arr[:, :, 1]), np.mean(arr[:, :, 2])]])
    img_batch = np.expand_dims(arr, axis=0)

    probas = hybrid_model.predict([img_batch, rgb_feat], verbose=0)[0]
    warna_pred = label_warna[np.argmax(probas)]
    return warna_pred, dict(zip(label_warna, probas))


# =====================================================
# LOAD CITRA KNOWN
# =====================================================
known_folder = "static/dataset_citra"

known_embeddings = []
known_colors = []
known_file_list = []

for fn in dataset['Tempat']:
    img_path = os.path.join(known_folder, fn)

    if os.path.exists(img_path):
        arr = preprocess_image(img_path)

        emb = extract_embedding(img_path)
        col = extract_color_feature(arr)

        known_embeddings.append(emb)
        known_colors.append(col)
        known_file_list.append(fn)

known_embeddings = np.array(known_embeddings)
known_colors = np.array(known_colors)

dataset = dataset[dataset['Tempat'].isin(known_file_list)].reset_index(drop=True)


# =====================================================
# WEIGHTED SIMILARITY + PERSENTASE KEMIRIPAN
# =====================================================
def weighted_similarity_filtered(emb_uji, col_uji, warna_pred, w1=0.75, w2=0.25, top_k=3):

    mask = dataset["Warna"] == warna_pred
    filtered_embeddings = known_embeddings[mask]
    filtered_colors = known_colors[mask]
    filtered_dataset = dataset[mask].reset_index(drop=True)

    if len(filtered_dataset) == 0:
        filtered_embeddings = known_embeddings
        filtered_colors = known_colors
        filtered_dataset = dataset.copy()

    d_cnn = pairwise_distances([emb_uji], filtered_embeddings, metric="euclidean")[0]
    d_col = pairwise_distances([col_uji], filtered_colors, metric="euclidean")[0]

    d_total = w1 * d_cnn + w2 * d_col

    sim = 1 - (d_total / d_total.max())
    sim_percent = sim * 100

    idx = np.argsort(d_total)[:top_k]

    result = filtered_dataset.iloc[idx].copy()
    result["Kemiripan"] = sim_percent[idx]

    return result


# =====================================================
# ROUTE
# =====================================================
@app.route("/", methods=["GET", "POST"])
def index():

    hasil = False
    context = {}

    if request.method == "POST":
        file = request.files["gambar"]
        if file:

            filename = file.filename
            upload_folder = os.path.join('static', 'uploaded')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            # Prediksi warna CNN (tetap dipakai)
            warna_pred, probas = prediksi_warna(file_path)
            warna_hex = warna_hex_mapping.get(warna_pred, "#000")

            # Ekstraksi palet warna (fitur baru)
            palette = get_dominant_colors(file_path, k=3)

            arr_uji = preprocess_image(file_path)
            emb_uji = extract_embedding(file_path)
            col_uji = extract_color_feature(arr_uji)

            kolong_mirip = weighted_similarity_filtered(emb_uji, col_uji, warna_pred, top_k=3)
            kolong_table = kolong_mirip.to_dict(orient="records")

            hasil = True
            context = {
                "image_path": url_for('static', filename=f'uploaded/{filename}'),
                "warna_prediksi": warna_pred,
                "warna_hex": warna_hex,
                "palette": palette,
                "kolong_table": kolong_table
            }

    return render_template("index.html", hasil=hasil, **context)


if __name__ == "__main__":
    app.run(debug=True)
