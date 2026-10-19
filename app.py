"""
app.py
============================================================
Web dashboard prediksi harga rumah.

Cara menjalankan:
    streamlit run app.py

Isi file ini HANYA urusan tampilan. Semua yang berhubungan
dengan data dan machine learning ada di model.py.
============================================================
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import model as ml


# ============================================================
# PENGATURAN HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide",
)

# Kurs dipakai hanya untuk tampilan. Dataset aslinya dalam dolar.
KURS_USD_IDR = 16_000

# Warna tema
WARNA = "#2563eb"


# ============================================================
# MEMUAT DATA & MODEL
# ============================================================
# Penting: tanpa @st.cache, model akan dilatih ulang tiap kali
# kamu klik atau geser slider. Dengan @st.cache, cukup sekali saja.

@st.cache_data(show_spinner="Memuat dataset...")
def ambil_data():
    """Mengambil dataset yang sudah dibersihkan."""
    return ml.bersihkan_data(ml.muat_data())


@st.cache_resource(show_spinner="Melatih model machine learning...")
def ambil_model():
    """Melatih model Linear Regression & Random Forest."""
    return ml.latih_model(ambil_data()[0])


data, laporan_bersih = ambil_data()
model = ambil_model()

# Hitung akurasi kedua model
akurasi_linear = ml.nilai_akurasi(model["linear"], model["X_uji"], model["y_uji"])
akurasi_forest = ml.nilai_akurasi(model["forest"], model["X_uji"], model["y_uji"])


# ============================================================
# FUNGSI BANTUAN (utility)
# ============================================================
def ke_rupiah(harga_dolar, ringkas=False):
    """Mengubah angka dolar jadi Rupiah, biar gampang dibaca.

    ringkas=True  -> dipakai untuk kartu metrik, hasilnya pendek
                     supaya tidak terpotong, contoh "Rp 312 Jt"
    ringkas=False -> dipakai untuk tabel, contoh "Rp 312.450.000"
    """
    angka = harga_dolar * KURS_USD_IDR

    if not ringkas:
        return f"Rp {angka:,.0f}".replace(",", ".")

    # Versi singkat: ribuan, juta, atau miliar
    if angka >= 1_000_000_000:
        return f"Rp {angka / 1_000_000_000:,.1f} M".replace(",", ".")
    if angka >= 1_000_000:
        return f"Rp {angka / 1_000_000:,.0f} Jt".replace(",", ".")
    return f"Rp {angka:,.0f}".replace(",", ".")


def ke_dolar(harga_dolar):
    """Menulis angka dolar dengan pemisah ribuan."""
    return f"${harga_dolar:,.0f}"


def judul_grafik(teks):
    """Judul grafik dengan gaya yang sama."""
    st.markdown(f"**{teks}**")


def kartu_nilai(nilai, label, warna=WARNA):
    """
    Membuat kotak angka ala "kartu statistik".

    st.metric bawaan Streamlit memotong teks kalau kolomnya
    sempit. Karena itu kita buat sendiri pakai HTML supaya
    angka panjang seperti "Rp 3,3 M" tetap utuh.
    """
    st.markdown(
        f"""
        <div style="
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 14px 16px;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(15,23,42,0.06);
            border-left: 4px solid {warna};
        ">
            <div style="
                font-size: 12px;
                color: #64748b;
                margin-bottom: 6px;
                font-weight: 500;
                line-height: 1.3;
            ">{label}</div>
            <div style="
                font-size: 22px;
                font-weight: 700;
                color: {warna};
                line-height: 1.1;
                white-space: nowrap;
            ">{nilai}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR (MENU SAMPING)
# ============================================================
with st.sidebar:
    st.header("⚙️ Filter Data")

    st.caption("Ubah filter ini, lalu semua grafik & angka ikut berubah.")

    st.subheader("Pendapatan")
    batas_pendapatan = (
        float(data["Pendapatan"].min()),
        float(data["Pendapatan"].max()),
    )
    pilih_pendapatan = st.slider(
        "Rentang pendapatan daerah",
        min_value=batas_pendapatan[0],
        max_value=batas_pendapatan[1],
        value=(1.0, 12.0),
        step=0.5,
        format="%.1f",
    )

    st.subheader("Umur Rumah")
    pilih_umur = st.slider(
        "Rentang umur rumah (tahun)",
        min_value=int(data["UmurRumah"].min()),
        max_value=int(data["UmurRumah"].max()),
        value=(1, 52),
    )

    st.divider()

    st.subheader("Tentang Data")
    st.markdown(
        f"""
        **California Housing**  
        Dataset resmi dari scikit-learn.

        - **{len(data):,}** baris data  
          (dari 20.640 baris awal, sisanya dibuang saat pembersihan)
        - **8** fitur, 1 target
        - Menggambarkan kelompok rumah di negara bagian
          California, Amerika Serikat, sekitar tahun 1990
        """
    )


# ============================================================
# MEMFILTER DATA
# ============================================================
# Filter dipakai di seluruh halaman
terfilter = data[
    (data["Pendapatan"] >= pilih_pendapatan[0])
    & (data["Pendapatan"] <= pilih_pendapatan[1])
    & (data["UmurRumah"] >= pilih_umur[0])
    & (data["UmurRumah"] <= pilih_umur[1])
]

# Kalau filter terlalu sempit, datanya habis. Kita kasih
# peringatan daripada bikin halaman error.
if terfilter.empty:
    st.warning("Filter yang dipilih terlalu sempit, tidak ada data. "
               "Coba lebarkan lagi filter di menu samping.")
    st.stop()


# ============================================================
# JUDUL HALAMAN
# ============================================================
st.title("🏠 Prediksi Harga Rumah")
st.markdown(
    """
    Aplikasi analisis data dan machine learning untuk memperkirakan
    harga rumah berdasarkan kondisi rumah dan pendapatan daerahnya.

    **Cara kerja aplikasinya:**
    1. Dataset dibersihkan dari baris yang tidak valid
    2. Model dilatih pada 80% data
    3. Akurasi diuji pada 20% data yang belum pernah dilihat
    4. Kamu bisa mencoba prediksi sendiri lewat panel di bawah
    """
)

st.divider()


# ============================================================
# KARTU RINGKASAN (METRIC)
# ============================================================
jumlah_data = len(terfilter)
harga_rata = terfilter["Harga"].mean() * 100_000
harga_median = terfilter["Harga"].median() * 100_000
harga_min = terfilter["Harga"].min() * 100_000
harga_max = terfilter["Harga"].max() * 100_000

kolom1, kolom2, kolom3, kolom4 = st.columns(4)

with kolom1:
    kartu_nilai(f"{jumlah_data:,}", "Jumlah Data", "#0f172a")

with kolom2:
    kartu_nilai(ke_rupiah(harga_rata, ringkas=True), "Rata-rata Harga")

with kolom3:
    kartu_nilai(ke_rupiah(harga_min, ringkas=True), "Harga Paling Murah", "#16a34a")

with kolom4:
    kartu_nilai(ke_rupiah(harga_max, ringkas=True), "Harga Paling Mahal", "#dc2626")

st.caption(
    f"Harga median {ke_dolar(harga_median)}  |  "
    "1 dolar dianggap Rp 16.000 (kurs asumsi, untuk memudahkan pembacaan). "
    "Pada kartu di atas: **J** = juta, **M** = miliar."
)

st.divider()


# ============================================================
# TAB: RINGKASAN, PREDIKSI, MODEL
# ============================================================
tab_ringkasan, tab_prediksi, tab_model = st.tabs(
    ["📊 Analisis Data", "🔮 Prediksi Harga", "🤖 Perbandingan Model"]
)


# ------------------------------------------------------------
# TAB 1: ANALISIS DATA
# ------------------------------------------------------------
with tab_ringkasan:
    st.subheader("1. Sebaran Harga Rumah")
    st.caption("Grafik ini menunjukkan sebaran harga rumah yang tersedia.")

    grafik1, grafik2 = st.columns(2)

    with grafik1:
        judul_grafik("Histogram Harga")
        st.plotly_chart(
            px.histogram(
                terfilter,
                x="Harga",
                nbins=50,
                labels={"Harga": "Harga (dolar)"},
                color_discrete_sequence=[WARNA],
            ),
            width="stretch",
        )
        st.caption(
            "Kalau bentuknya seperti Gunung (puncak di tengah), "
            "artinya sebagian besar rumah harganya sedang saja."
        )

    with grafik2:
        judul_grafik("Hubungan Pendapatan vs Harga")
        st.plotly_chart(
            px.scatter(
                terfilter,
                x="Pendapatan",
                y="Harga",
                labels={
                    "Pendapatan": "Pendapatan ($10.000)",
                    "Harga": "Harga ($100.000)",
                },
                color_discrete_sequence=[WARNA],
                opacity=0.5,
            ),
            width="stretch",
        )
        st.caption(
            "Semakin ke kanan (pendapatan makin tinggi), "
            "semakin ke atas (harga makin tinggi)."
        )

    st.divider()
    st.subheader("2. Pengaruh Setiap Fitur")

    kolom_a, kolom_b = st.columns(2)

    with kolom_a:
        judul_grafik("Harga Berdasarkan Kuartil Pendapatan")
        # Bagi data menjadi 4 kelompok berdasarkan pendapatan
        Salin = terfilter.copy()
        Salin["Kelompok"] = pd.qcut(
            Salin["Pendapatan"],
            q=4,
            labels=["1. Rendah", "2. Sedang Bawah", "3. Sedang Atas", "4. Tinggi"],
        )
        st.plotly_chart(
            px.box(
                Salin,
                x="Kelompok",
                y="Harga",
                color="Kelompok",
                labels={"Harga": "Harga ($100.000)"},
                color_discrete_sequence=["#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8"],
            ),
            width="stretch",
        )
        st.caption(
            "Kotak yang makin tinggi artinya harga di kelompok itu "
            "lebih beragam."
        )

    with kolom_b:
        judul_grafik("Harga Berdasarkan Umur Rumah")
        st.plotly_chart(
            px.box(
                terfilter,
                x="UmurRumah",
                y="Harga",
                labels={
                    "UmurRumah": "Umur rumah (tahun)",
                    "Harga": "Harga ($100.000)",
                },
                color_discrete_sequence=[WARNA],
            ),
            width="stretch",
        )
        st.caption(
            "Rumah yang lebih tua biasanya punya harga lebih rendah."
        )

    st.divider()
    st.subheader("3. Heatmap Korelasi")
    st.caption(
        "Korelasi bernilai -1 sampai +1. Makin dekat ke +1 atau -1, "
        "semakin kuat hubungannya. Warna merah = naik, biru = turun."
    )

    kolom_korelasi = ml.FITUR + ["Lintang", "Bujur", ml.TARGET]
    matriks_korelasi = terfilter[kolom_korelasi].corr()

    # Ganti nama kolom supaya enak dibaca
    Pretty_ = {
        "Pendapatan": "Pendapatan",
        "UmurRumah": "Umur",
        "RataRumah": "Kamar",
        "RataPenghuni": "Penghuni",
        "RataKamarTidur": "Kamar tidur",
        "Lintang": "Lintang",
        "Bujur": "Bujur",
        "Harga": "HARGA",
    }
    matriks_korelasi = matriks_korelasi.rename(index=Pretty_, columns=Pretty_)

    st.plotly_chart(
        px.imshow(
            matriks_korelasi,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        ),
        width="stretch",
    )

    st.divider()
    st.subheader("4. Contoh Isi Data")
    st.caption("Lima baris pertama dari data yang sedang difilter.")

    contoh = terfilter[["Pendapatan", "UmurRumah", "RataRumah",
                        "RataPenghuni", "RataKamarTidur", "Harga"]].head(5)
    st.dataframe(contoh, width="stretch", hide_index=True)

    st.divider()
    st.subheader("5. Proses Pembersihan Data")
    st.caption(
        "Beberapa baris tidak masuk akal (misalnya rumah tanpa kamar tidur) "
        "dan dibuang agar tidak mengacaukan model."
    )

    total_buang = int(laporan_bersih["Baris Dibuang"].sum())

    kolom_b1, kolom_b2 = st.columns(2)
    with kolom_b1:
        kartu_nilai(f"{total_buang:,}", "Baris dibuang", "#dc2626")
    with kolom_b2:
        kartu_nilai(f"{len(data):,}", "Baris tersisa", "#16a34a")

    st.dataframe(
        laporan_bersih,
        width="stretch",
        hide_index=True,
    )


# ------------------------------------------------------------
# TAB 2: PREDIKSI HARGA
# ------------------------------------------------------------
with tab_prediksi:
    st.subheader("Coba Prediksi Harga Rumah")
    st.caption(
        "Geser slider di bawah sesuai kondisi rumah yang kamubayangkan, "
        "lalu klik tombol prediksi."
    )

    with st.form("form_prediksi"):
        kolom_f1, kolom_f2 = st.columns(2)

        with kolom_f1:
            input_pendapatan = st.slider(
                "Pendapatan median daerah ($10.000)",
                min_value=0.5,
                max_value=15.0,
                value=3.0,
                step=0.5,
                help=ml.KETERANGAN["Pendapatan"],
            )

            input_umur = st.slider(
                "Umur rumah (tahun)",
                min_value=1,
                max_value=52,
                value=20,
                help=ml.KETERANGAN["UmurRumah"],
            )

            input_kamar = st.slider(
                "Rata-rata jumlah kamar",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.1,
                help=ml.KETERANGAN["RataRumah"],
            )

        with kolom_f2:
            input_penghuni = st.slider(
                "Rata-rata jumlah penghuni",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
                help=ml.KETERANGAN["RataPenghuni"],
            )

            input_kamar_tidur = st.slider(
                "Rata-rata jumlah kamar tidur",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help=ml.KETERANGAN["RataKamarTidur"],
            )

        submitted = st.form_submit_button("🔮 Prediksi Harga", type="primary")

    # ----------------------------------------------------
    # Tombol baru diproses kalau diklik
    # ----------------------------------------------------
    if submitted:
        # Bikin 1 baris data baru sesuai input pengguna
        input_baru = pd.DataFrame([{
            "Pendapatan": input_pendapatan,
            "UmurRumah": input_umur,
            "RataRumah": input_kamar,
            "RataPenghuni": input_penghuni,
            "RataKamarTidur": input_kamar_tidur,
        }])

        pred_linear = model["linear"].predict(input_baru)[0] * 100_000
        pred_forest = model["forest"].predict(input_baru)[0] * 100_000

        st.divider()
        st.markdown("### Hasil Prediksi")

        kolom_h1, kolom_h2 = st.columns(2)

        with kolom_h1:
            kartu_nilai(
                ke_rupiah(pred_linear, ringkas=True),
                f"Linear Regression — {ke_dolar(pred_linear)}",
            )

        with kolom_h2:
            kartu_nilai(
                ke_rupiah(pred_forest, ringkas=True),
                f"Random Forest — {ke_dolar(pred_forest)}",
                "#16a34a",
            )

        # Bandingkan dengan data asli yang paling mirip
        mirip = data[
            (data["Pendapatan"] >= input_pendapatan - 1)
            & (data["Pendapatan"] <= input_pendapatan + 1)
        ]

        st.markdown("---")

        if not mirip.empty:
            st.markdown("#### Pembanding dari Data Asli")
            st.caption(
                "Data asli yang pendapatan daerahnya paling mirip "
                f"dengan input kamu ({len(mirip)} baris). Bandingkan dengan "
                "hasil prediksi model di atas: kalau angkanya jauh berbeda, "
                "berarti input kamu berada di luar rentang data yang ada."
            )

            pembanding = pd.DataFrame({
                "Harga Minimum": [mirip["Harga"].min() * 100_000],
                "Rata-rata": [mirip["Harga"].mean() * 100_000],
                "Harga Maximum": [mirip["Harga"].max() * 100_000],
            })

            st.dataframe(
                pembanding.map(ke_rupiah),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(
                "Tidak ada data asli yang cukup mirip dengan input kamu. "
                "Coba naikkan sedikit nilai pendapatan daerah."
            )

        st.markdown("#### Grafik Harga")

        batas_atas = float(np.percentile(data["Harga"], 99)) * 100_000

        data_grafik = pd.DataFrame({
            "Sumber": [
                "Prediksi Linear Regression",
                "Prediksi Random Forest",
                "Rata-rata data asli",
            ],
            "Harga": [
                pred_linear,
                pred_forest,
                mirip["Harga"].mean() * 100_000 if not mirip.empty else np.nan,
            ],
        })

        st.plotly_chart(
            px.bar(
                data_grafik,
                x="Sumber",
                y="Harga",
                text="Harga",
                labels={"Harga": "Harga (dolar)", "Sumber": ""},
                color_discrete_sequence=[WARNA, "#f59e0b", "#94a3b8"],
            ),
            width="stretch",
        )

        st.caption(
            f"Untuk perbandingan, 99% rumah di dataset ini berharga "
            f"di bawah {ke_dolar(batas_atas)}."
        )

    else:
        st.info("Isi slider di atas lalu klik tombol **Prediksi Harga**.")


# ------------------------------------------------------------
# TAB 3: PERBANDINGAN MODEL
# ------------------------------------------------------------
with tab_model:
    st.subheader("Model 1: Linear Regression")
    st.caption(
        "Model paling sederhana. Dugaannya: "
        "semakin besar setiap fitur, semakin besar harganya, "
        "dan hubungannya lurus (garis lurus)."
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("R²", f"{akurasi_linear['R2']:.3f}")
    k2.metric("MAE", ke_dolar(akurasi_linear["MAE"] * 100_000))
    k3.metric("RMSE", ke_dolar(akurasi_linear["RMSE"] * 100_000))

    st.markdown("#### Pengaruh tiap fitur")
    st.caption(
        "Semakin besar angkanya, semakin besar pengaruhnya. "
        "Angka negatif berarti: fitur ini justru menurunkan harga."
    )

    penting_linear = ml.importance_fitur(model["linear"], "linear")
    st.plotly_chart(
        px.bar(
            penting_linear,
            x="Pengaruh",
            y="Fitur",
            orientation="h",
            labels={"Pengaruh": "Koefisien", "Fitur": ""},
            color_discrete_sequence=[WARNA],
        ),
        width="stretch",
    )

    st.divider()
    st.subheader("Model 2: Random Forest")
    st.caption(
        "Model yang lebih rumit secara komputasi tapi biasanya lebih akurat. "
        "Cara kerjanya: membangun 100 pohon keputusan, lalu "
        "mengambil rata-rata jawaban semua pohon."
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("R²", f"{akurasi_forest['R2']:.3f}")
    k2.metric("MAE", ke_dolar(akurasi_forest["MAE"] * 100_000))
    k3.metric("RMSE", ke_dolar(akurasi_forest["RMSE"] * 100_000))

    st.markdown("#### Pengaruh tiap fitur")
    st.caption(
        "Di sini semua nilai positif. Makin besar berarti "
        "fitur itu makin sering dipakai model untuk mengambil keputusan."
    )

    penting_forest = ml.importance_fitur(model["forest"], "forest")
    st.plotly_chart(
        px.bar(
            penting_forest,
            x="Pengaruh",
            y="Fitur",
            orientation="h",
            labels={"Pengaruh": "Pengaruh (%)", "Fitur": ""},
            color_discrete_sequence=["#16a34a"],
        ),
        width="stretch",
    )

    st.divider()
    st.subheader("Perbandingan Langsung")

    tabel_banding = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "R²": [akurasi_linear["R2"], akurasi_forest["R2"]],
        "MAE ($)": [akurasi_linear["MAE"] * 100_000,
                    akurasi_forest["MAE"] * 100_000],
        "RMSE ($)": [akurasi_linear["RMSE"] * 100_000,
                     akurasi_forest["RMSE"] * 100_000],
    })

    st.dataframe(
        tabel_banding.style.format({
            "R²": "{:.3f}",
            "MAE ($)": "${:,.0f}",
            "RMSE ($)": "${:,.0f}",
        }),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Prediksi vs Data Asli")
    st.caption(
        "Semakin titik-titiknya menempel pada garis diagonal, "
        "semakin akurat modelnya."
    )

    X_uji = model["X_uji"]
    y_uji = model["y_uji"]

    banding_data = pd.DataFrame({
        "Harga Asli ($)": y_uji.values * 100_000,
        "Prediksi Random Forest ($)": model["forest"].predict(X_uji) * 100_000,
    })

    st.plotly_chart(
        px.scatter(
            banding_data,
            x="Harga Asli ($)",
            y="Prediksi Random Forest ($)",
            opacity=0.4,
            color_discrete_sequence=["#16a34a"],
            labels={
                "Harga Asli ($)": "Harga asli (dolar)",
                "Prediksi Random Forest ($)": "Prediksi model (dolar)",
            },
        ),
        width="stretch",
    )

    st.info(
        "Kesimpulan: "
        f"Random Forest mencapai R² {akurasi_forest['R2']:.2f}, "
        "sedangkan Linear Regression hanya "
        f"{akurasi_linear['R2']:.2f}. "
        "Artinya Random Forest lebih tepat dipakai untuk "
        "prediksi harga rumah."
    )


# ============================================================
# PENUTUP HALAMAN
# ============================================================
st.divider()

with st.expander("📚 Metodologi & Penjelasan"):
    st.markdown(
        """
        #### Dataset
        California Housing dari scikit-learn, berisi 20.640 baris
        yang menggambarkan kondisi rumah di negara bagian
        California, Amerika Serikat, sekitar tahun 1990.

        #### Data Cleaning
        Pembersihan dilakukan dalam dua tahap.

        **Tahap 1 — aturan fisik.** Baris yang isinya tidak mungkin
        terjadi dibuang: rumah dengan kurang dari 1 kamar, lebih dari 20 kamar, lebih dari 6 kamar tidur,
        dan lebih dari 8 penghuni.

        **Tahap 2 — outlier dengan metode IQR.** Dihitung
        batas bawah = Q1 - 1,5 x IQR dan batas atas = Q3 + 1,5 x IQR,
        lalu semua nilai di luar batas itu dibuang. Metode ini
        diterapkan pada kolom jumlah penghuni.

        Catatan: kolom "RataKamarTidur" sengaja tidak ikut dibersihkan,
        meskipun ada nilainya di bawah 1. Kolom itu berisi rata-rata
        jumlah kamar tidur per rumah dalam satu wilayah, sehingga di
        wilayah padat penduduk nilai rata-ratanya memang bisa di bawah 1.
        Itu wajar, bukan kesalahan data.

        #### Pembagian Data
        Data dibagi 80% untuk melatih model dan 20% untuk menguji.
        Model hanya dinilai dengan data uji yang belum pernah
        dilihat sebelumnya, supaya hasilnya jujur.

        #### Metrik
        - **R²** : berapa persen variasi harga yang bisa dijelaskan
          oleh model. Makin dekat 1 makin bagus.
        - **MAE** : rata-rata besar kesalahan prediksi, dalam dolar.
        - **RMSE** : sama seperti MAE, tetapi kesalahan yang sangat
          besar dihukum lebih berat.

        #### library yang dipakai
        - `streamlit` — membuat web app
        - `pandas` — mengolah tabel data
        - `scikit-learn` — membuat machine learning model
        - `plotly` — membuat grafik interaktif
        """
    )
