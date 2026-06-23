import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import tempfile
import zipfile
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Klasifikasi Penyakit Daun Padi", layout="wide")

st.title("🌾 Klasifikasi Penyakit Daun Padi")
st.markdown("**SVM + CLAHE | Deteksi Penyakit: Bacterial Blight, Brown Spot, Leaf Smut**")

# ========== FUNCTIONS ==========

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = lab.copy()
    enhanced_lab[:, :, 0] = enhanced_l
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_image

def preprocess_image(image_path, target_size=(224, 224), use_clahe=True):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, target_size)
    if use_clahe:
        processed = apply_clahe(resized)
    else:
        processed = resized
    return processed

def augment_image(image):
    augmented = [image]
    flipped = cv2.flip(image, 1)
    augmented.append(flipped)
    for angle in [-30, -15, 15, 30]:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)
    for alpha in [0.7, 0.85, 1.15, 1.3]:
        bright = np.clip(image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        augmented.append(bright)
    for scale in [0.85, 1.15]:
        h, w = image.shape[:2]
        if scale < 1.0:
            ch, cw = int(h * scale), int(w * scale)
            y1, x1 = (h - ch) // 2, (w - cw) // 2
            cropped = image[y1:y1+ch, x1:x1+cw]
            zoomed = cv2.resize(cropped, (w, h))
        else:
            zoomed = cv2.resize(image, (int(w * scale), int(h * scale)))
            y1 = (zoomed.shape[0] - h) // 2
            x1 = (zoomed.shape[1] - w) // 2
            zoomed = zoomed[y1:y1+h, x1:x1+w]
        augmented.append(zoomed)
    return augmented

def extract_color_features(image):
    features = []
    for i in range(3):
        channel = image[:, :, i]
        features.extend([np.mean(channel), np.std(channel), np.min(channel), np.max(channel)])
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for i in range(3):
        channel = hsv[:, :, i]
        features.extend([np.mean(channel), np.std(channel)])
    return np.array(features)

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (64, 64))
    gray_quantized = (gray / 32).astype(np.uint8)
    glcm = graycomatrix(gray_quantized, distances=[1, 2], angles=[0, np.pi/4, np.pi/2],
                       levels=8, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        features.append(graycoprops(glcm, prop).mean())
    return np.array(features)

def extract_features(image, use_clahe=True):
    if use_clahe:
        image = apply_clahe(image)
    color_features = extract_color_features(image)
    texture_features = extract_texture_features(image)
    return np.concatenate([color_features, texture_features])

def load_dataset(dataset_path, use_clahe=True, target_size=(224, 224), apply_augmentation=False):
    features_list, labels_list = [], []
    class_dirs = sorted([d for d in os.listdir(dataset_path)
                         if os.path.isdir(os.path.join(dataset_path, d))])
    if not class_dirs:
        raise ValueError(f'Tidak ada folder kelas ditemukan')
    class_names = class_dirs
    progress_bar = st.progress(0)
    status_text = st.empty()
    for label, class_name in enumerate(class_dirs):
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for idx, img_file in enumerate(image_files):
            try:
                img_path = os.path.join(class_path, img_file)
                image = preprocess_image(img_path, target_size, use_clahe)
                if image is None:
                    continue
                images_to_process = augment_image(image) if apply_augmentation else [image]
                for img in images_to_process:
                    feat = extract_features(img, use_clahe=False)
                    features_list.append(feat)
                    labels_list.append(label)
            except:
                continue
        progress = (label + 1) / len(class_dirs)
        progress_bar.progress(progress)
        status_text.text(f"Loading {class_name}...")
    progress_bar.empty()
    status_text.empty()
    X = np.array(features_list)
    y = np.array(labels_list)
    return X, y, class_names

# ========== SIDEBAR ==========

st.sidebar.header("Dataset")
dataset_option = st.sidebar.radio("Sumber Dataset", ["Upload ZIP Dataset", "Gunakan Path Folder"])

dataset_path = None
if dataset_option == "Upload ZIP Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload file ZIP dataset", type=["zip"])
    if uploaded_file:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "dataset.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)
            st.sidebar.success("Dataset berhasil diekstrak!")
            dataset_path = tmpdir
else:
    dataset_path = st.sidebar.text_input("Path folder dataset", placeholder="/path/to/dataset")

st.sidebar.header("Pengaturan")
use_clahe = st.sidebar.checkbox("Gunakan CLAHE", value=True)
apply_augment = st.sidebar.checkbox("Aktifkan Augmentasi Data", value=False)
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

st.sidebar.header("Hyperparameter SVM")
kernel_option = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
c_value = st.sidebar.select_slider("C", options=[0.1, 1, 10, 100], value=10)
gamma_value = st.sidebar.selectbox("Gamma", ["scale", "auto", 0.001, 0.01, 0.1])

# ========== MAIN TABS ==========

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Informasi", "🔧 Preprocessing & Augmentasi", "📊 Training & Evaluasi",
    "🔍 Prediksi Gambar", "📈 Perbandingan"
])

with tab1:
    st.header("Informasi Proyek")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Klasifikasi Penyakit Daun Padi** menggunakan **Support Vector Machine (SVM)** 
        dengan **CLAHE (Contrast Limited Adaptive Histogram Equalization)** untuk preprocessing.

        **Kelas Penyakit:**
        - **Bacterial leaf blight** – Infeksi bakteri *Xanthomonas oryzae*
        - **Brown spot** – Infeksi jamur *Helminthosporium oryzae*
        - **Leaf smut** – Infeksi jamur *Tilletia horrida*

        **Pipeline:**
        1. Preprocessing: Resize → CLAHE
        2. Augmentasi Data (opsional): flip, rotasi, brightness, zoom
        3. Ekstraksi Fitur: fitur warna (RGB + HSV) + fitur tekstur (GLCM)
        4. Klasifikasi: SVM dengan hyperparameter tuning
        """)
    with col2:
        st.info("Dataset: Rice Leaf Disease Dataset (Kaggle)\n120 citra (40/kelas)")

with tab2:
    st.header("Preprocessing & Augmentasi")
    st.markdown("Upload gambar untuk melihat efek CLAHE dan augmentasi.")

    uploaded_img = st.file_uploader("Upload gambar daun padi", type=["jpg", "jpeg", "png"], key="preview")
    if uploaded_img:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, caption="Original", use_container_width=True)
        with col2:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            st.image(gray, caption="Grayscale", use_container_width=True, clamp=True)
        with col3:
            clahe_img = apply_clahe(img)
            st.image(clahe_img, caption="Setelah CLAHE", use_container_width=True)

        if st.button("Tampilkan Augmentasi"):
            aug_images = augment_image(img)
            titles = ['Original', 'H-Flip', 'Rot -30°', 'Rot -15°', 'Rot +15°', 'Rot +30°',
                      'Bright 70%', 'Bright 85%', 'Bright 115%', 'Bright 130%', 'Zoom In', 'Zoom Out']
            cols = st.columns(6)
            for i, (aug_img, title) in enumerate(zip(aug_images, titles)):
                with cols[i % 6]:
                    st.image(aug_img, caption=title, use_container_width=True)

with tab3:
    st.header("Training & Evaluasi Model")

    if not dataset_path:
        st.warning("Silakan upload dataset atau masukkan path folder di sidebar.")
    else:
        if st.button("🚀 Mulai Training", type="primary"):
            with st.spinner("Loading dataset..."):
                X, y, class_names = load_dataset(
                    dataset_path, use_clahe=use_clahe, apply_augmentation=apply_augment
                )

            st.success(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
            st.write(f"**Kelas:** {class_names}")
            st.write(f"**Distribusi:** {np.bincount(y)}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            with st.spinner("Training SVM..."):
                svm_model = SVC(
                    C=c_value, kernel=kernel_option,
                    gamma=gamma_value, random_state=42, probability=True
                )
                svm_model.fit(X_train_scaled, y_train)

            y_pred = svm_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            st.metric("Akurasi", f"{accuracy*100:.2f}%")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report.style.format("{:.4f}"), use_container_width=True)

            with col2:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names, ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                st.pyplot(fig)
                plt.close()

            # Save model
            model_data = {
                'svm_model': svm_model,
                'scaler': scaler,
                'class_names': class_names,
                'accuracy': accuracy
            }
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                pickle.dump(model_data, f)
                model_path = f.name

            with open(model_path, "rb") as f:
                st.download_button(
                    "📥 Download Model (.pkl)",
                    f,
                    file_name="svm_clahe_model.pkl",
                    mime="application/octet-stream"
                )

            st.session_state['model_data'] = model_data
            st.session_state['class_names'] = class_names

with tab4:
    st.header("Prediksi Gambar")

    if 'model_data' not in st.session_state:
        st.warning("Silakan training model terlebih dahulu di tab Training.")
    else:
        model_data = st.session_state['model_data']
        svm_model = model_data['svm_model']
        scaler = model_data['scaler']
        class_names = model_data['class_names']

        uploaded_pred = st.file_uploader("Upload gambar untuk diprediksi", type=["jpg", "jpeg", "png"], key="predict")
        if uploaded_pred:
            file_bytes = np.asarray(bytearray(uploaded_pred.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))

            col1, col2 = st.columns(2)
            with col1:
                st.image(img_rgb, caption="Gambar Input", use_container_width=True)

            with col2:
                if use_clahe:
                    processed = apply_clahe(img_resized)
                else:
                    processed = img_resized
                features = extract_features(processed, use_clahe=False).reshape(1, -1)
                features_scaled = scaler.transform(features)
                pred = svm_model.predict(features_scaled)[0]
                proba = svm_model.predict_proba(features_scaled)[0]

                pred_class = class_names[pred]
                st.success(f"**Hasil Prediksi: {pred_class}**")
                st.write("**Probabilitas per kelas:**")
                for name, prob in zip(class_names, proba):
                    st.progress(float(prob))
                    st.write(f"{name}: {prob*100:.2f}%")

with tab5:
    st.header("Perbandingan CLAHE vs Non-CLAHE")

    if not dataset_path:
        st.warning("Silakan upload dataset di sidebar.")
    else:
        if st.button("Bandingkan CLAHE vs Non-CLAHE", type="primary"):
            with st.spinner("Evaluasi tanpa CLAHE..."):
                X_no_clahe, y_no_clahe, _ = load_dataset(
                    dataset_path, use_clahe=False, apply_augmentation=False
                )
                X_clahe, y_clahe, class_names = load_dataset(
                    dataset_path, use_clahe=True, apply_augmentation=False
                )

            def evaluate_pipeline(X, y):
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_te_scaled = scaler.transform(X_te)
                model = SVC(C=c_value, kernel=kernel_option, gamma=gamma_value, random_state=42)
                model.fit(X_tr_scaled, y_tr)
                y_pred = model.predict(X_te_scaled)
                return accuracy_score(y_te, y_pred)

            acc_no_clahe = evaluate_pipeline(X_no_clahe, y_no_clahe)
            acc_with_clahe = evaluate_pipeline(X_clahe, y_clahe)

            col1, col2, col3 = st.columns(3)
            col1.metric("Tanpa CLAHE", f"{acc_no_clahe*100:.2f}%")
            col2.metric("Dengan CLAHE", f"{acc_with_clahe*100:.2f}%")
            improvement = (acc_with_clahe - acc_no_clahe) * 100
            col3.metric("Peningkatan", f"{improvement:+.2f}%", delta=f"{improvement:+.2f}%")

            fig, ax = plt.subplots(figsize=(8, 6))
            methods = ['Tanpa CLAHE', 'Dengan CLAHE']
            accs = [acc_no_clahe, acc_with_clahe]
            colors = ['lightcoral', 'lightgreen']
            bars = ax.bar(methods, accs, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Akurasi')
            ax.set_title('Perbandingan Performa: Dengan vs Tanpa CLAHE')
            for bar, acc in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{acc*100:.2f}%', ha='center', fontsize=12, fontweight='bold')
            st.pyplot(fig)
            plt.close()

st.sidebar.markdown("---")
st.sidebar.info("**UTS RTI - Klasifikasi Penyakit Daun Padi**\nSVM + CLAHE + GLCM")
