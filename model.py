"""
model.py
============================================================
Di file ini kita melakukan semua pekerjaan DATA dan MACHINE
LEARNING:

    1. Memuat dataset
    2. Membersihkan (cleaning) data
    3. Melatih (training) model prediksi
    4. Menghitung seberapa akurat model

Kenapa dipisah dari app.py?
Supaya app.py hanya mengurus tampilan, dan semua logika
data bisa diuji sendiri tanpa harus membuka web.

Dataset yang dipakai: "California Housing"
- 20.640 baris
- 8 fitur (keterangan rumah & lokasi)
- 1 target (harga rumah)

Sumber: scikit-learn (sudah termasuk di dalam paketnya,
jadi tidak perlu download manual dari internet)
============================================================
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. DAFTAR KOLOM
# ============================================================
# Nama kolom asli dari dataset memakai Bahasa Inggris dan
# singkatan yang membingungkan (misal "MedInc"). Kita ganti
# jadi Bahasa Indonesia supaya gampang dibaca di web.

# Daftar fitur = hal-hal yang kemungkinan memengaruhi harga
FITUR = [
    "Pendapatan",
    "UmurRumah",
    "RataRumah",
    "RataPenghuni",
    "RataKamarMandi",
]

# Target = hal yang ingin kita prediksi
TARGET = "Harga"

# Nama kolom asli -> nama baru (Bahasa Indonesia)
GANTI_NAMA = {
    "MedInc": "Pendapatan",
    "HouseAge": "UmurRumah",
    "AveRooms": "RataRumah",
    "AveOccup": "RataPenghuni",
    "AveBedrms": "RataKamarMandi",
    "Latitude": "Lintang",
    "Longitude": "Bujur",
    "MedHouseVal": "Harga",
}

# Keterangan singkat tiap fitur (dipakai sebagai tooltip di web)
KETERANGAN = {
    "Pendapatan": "Pendapatan median inhabitants (satuan: $10.000)",
    "UmurRumah": "Umur median rumah (tahun)",
    "RataRumah": "Rata-rata jumlah kamar per rumah",
    "RataPenghuni": "Rata-rata jumlah penghuni per rumah",
    "RataKamarMandi": "Rata-rata jumlah kamar mandi",
    "Harga": "Harga median rumah (satuan: $100.000)",
}


# ============================================================
# 2. MEMUAT DATASET
# ============================================================
def muat_data() -> pd.DataFrame:
    """
    Mengambil dataset harga rumah California dari scikit-learn.

    Mengembalikan:
        DataFrame (tabel data) berisi data harga rumah.
    """
    # Ambil datanya. Cara ini (data + target) berlaku di semua
    # versi scikit-learn, jadi aman.
    mentah = fetch_california_housing()

    # Bikin tabel dari fitur-fitur
    df = pd.DataFrame(mentah.data, columns=mentah.feature_names)

    # Target ditambahkan manual sebagai kolom terakhir
    df["MedHouseVal"] = mentah.target

    # Ganti nama kolom ke Bahasa Indonesia
    df = df.rename(columns=GANTI_NAMA)

    return df


# ============================================================
# 3. MEMBERSIHKAN DATA (DATA CLEANING)
# ============================================================
def bersihkan_data(df: pd.DataFrame) -> tuple:
    """
    Membersihkan dataset dari baris yang tidak valid dan nilai
    ekstrem (outlier).

    Dua jenis pembersihan:

    1. Aturan fisik
      (authentik mustahil terjadi, jadi dibuang)
         - Rumah dengan 0 kamar
         - Rumah dengan lebih dari 20 kamar
         - Rumah dengan lebih dari 6 kamar mandi
         - Rumah dengan lebih dari 8 penghuni

    2. Metode IQR (Interquartile Range)
       Metrik standar untuk menemukan outlier:
         batas bawah = Q1 - 1,5 x IQR
         batas atas  = Q3 + 1,5 x IQR
       Semua nilai di luar batas itu dianggap ekstrem.
       Diterapkan pada kolom "RataPenghuni".

    Catatan penting:
       Angka "RataKamarMandi" itu RATA-RATA per rumah dalam satu
       wilayah, bukan jumlah kamar mandi satu rumah. Karena itu
       ada wilayah yang rata-ratanya di bawah 1 dan itu WAJAR,
       bukan data salah. Jadi angka itu tidak ikut dihapus.

    Mengembalikan:
        (data_bersih, laporan)
    """
    laporan = [{
        "Tahap": "Data awal (sebelum dibersihkan)",
        "Baris Dibuang": 0,
        "Sisa Baris": len(df),
    }]

    # ---------- Bagian 1: aturan fisik ----------
    aturan_fisik = [
        ("Rumah minimal 1 kamar", "RataRumah", "ge", 1),
        ("Rumah maksimal 20 kamar", "RataRumah", "le", 20),
        ("Kamar mandi maksimal 6 per rumah", "RataKamarMandi", "le", 6),
        ("Penghuni maksimal 8 per rumah", "RataPenghuni", "le", 8),
    ]

    for keterangan, kolom, operator, batas in aturan_fisik:
        sebelum = len(df)

        if operator == "ge":
            df = df[df[kolom] >= batas]
        else:
            df = df[df[kolom] <= batas]

        laporan.append({
            "Tahap": keterangan,
            "Baris Dibuang": sebelum - len(df),
            "Sisa Baris": len(df),
        })

    # ---------- Bagian 2: outlier metode IQR ----------
    kolom_iqr = "RataPenghuni"
    q1 = df[kolom_iqr].quantile(0.25)
    q3 = df[kolom_iqr].quantile(0.75)
    iqr = q3 - q1
    batas_bawah = q1 - 1.5 * iqr
    batas_atas = q3 + 1.5 * iqr

    sebelum = len(df)
    df = df[(df[kolom_iqr] >= batas_bawah) & (df[kolom_iqr] <= batas_atas)]

    laporan.append({
        "Tahap": "Buang outlier RataPenghuni (IQR)",
        "Baris Dibuang": sebelum - len(df),
        "Sisa Baris": len(df),
    })

    return df.reset_index(drop=True), pd.DataFrame(laporan)


# ============================================================
# 4. MELATIH MODEL (MODEL TRAINING)
# ============================================================
def latih_model(df: pd.DataFrame) -> dict:
    """
    Membagi data menjadi data latih & data uji, lalu melatih
    DUA model supaya bisa dibandingkan:

        a) Linear Regression -> cara paling sederhana
        b) Random Forest     -> cara yang lebih rumit

    Kenapa dibagi dua?
    Supaya kita tahu seberapa akurat model saat melihat data
    yang BELUM PERNAH dilihat sebelumnya. Ini namanya test set.

    Mengembalikan:
        Dictionary berisi model + data latih/uji.
    """
    # Pisahkan kolom keterangan (X) dan kolom harga (y)
    X = df[FITUR]
    y = df[TARGET]

    # Bagi data: 80% untuk melatih, 20% untuk menguji.
    # random_state=42 supaya hasilnya selalu sama setiap dijalankan.
    X_latih, X_uji, y_latih, y_uji = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # a) Model Linear Regression
    linear = LinearRegression()
    linear.fit(X_latih, y_latih)

    # b) Model Random Forest (100 pohon keputusan)
    forest = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(X_latih, y_latih)

    return {
        "linear": linear,
        "forest": forest,
        "X_latih": X_latih,
        "X_uji": X_uji,
        "y_latih": y_latih,
        "y_uji": y_uji,
    }


# ============================================================
# 5. MENGUKUR AKURASI MODEL (EVALUATION)
# ============================================================
def nilai_akurasi(model, X_uji, y_uji) -> dict:
    """
    Menghitung seberapa bagus model dalam menebak harga.

    Mengembalikan 3 angka:
        R2   -> 0 sampai 1. Makin dekat 1, makin bagus.
                (0.70 artinya model menjelaskan 70% variasi harga)
        MAE  -> rata-rata salah prediksi, dalam $/100.000
        RMSE -> sama seperti MAE, tapi error besar dihukum
                lebih berat. Makin kecil makin bagus.
    """
    prediksi = model.predict(X_uji)

    return {
        "R2": r2_score(y_uji, prediksi),
        "MAE": mean_absolute_error(y_uji, prediksi),
        "RMSE": mean_squared_error(y_uji, prediksi) ** 0.5,
    }


# ============================================================
# 6. MELIHAT FITUR YANG PALING BERPENGARUH
# ============================================================
def importance_fitur(model, jenis: str) -> pd.DataFrame:
    """
    Melihat fitur mana yang paling berpengaruh terhadap harga.

    - Linear Regression : memakai koefisien (bisa bernilai minus)
    - Random Forest     : memakai feature importance (selalu plus)

    Mengembalikan:
        DataFrame 2 kolom (nama fitur + nilai), diurutkan dari
        yang paling berpengaruh.
    """
    if jenis == "linear":
        # Pada Linear Regression, besar kecil koefisien menunjukkan
        # seberapa kuat pengaruh fitur terhadap harga.
        nilai = model.coef_
    else:
        # Pada Random Forest, feature importance menunjukkan
        # seberapa sering fitur itu dipakai saat membuat keputusan.
        nilai = model.feature_importances_

    hasil = pd.DataFrame({
        "Fitur": FITUR,
        "Pengaruh": nilai,
    })

    return hasil.sort_values("Pengaruh", ascending=False).reset_index(drop=True)
