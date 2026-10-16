# 🏠 Prediksi Harga Rumah — Dashboard Analisis Data & Machine Learning

Aplikasi web untuk menganalisis data harga rumah dan memperkirakan
harga rumah berdasarkan kondisi rumah serta pendapatan daerahnya.

Dibuat dengan **Python + Streamlit**, lalu bisa di-deploy ke internet
sehingga bisa dibuka lewat browser tanpa harus install apa pun.

---

## 📋 Daftar Isi

1. [Isi Proyek](#-isi-proyek)
2. [Cara Menjalankan](#-cara-menjalankan)
3. [Penjelasan File](#-penjelasan-file)
4. [Cara Kerja Aplikasinya](#-cara-kerja-aplikasinya)
5. [Cara Deploy ke Internet](#-cara-deploy-ke-internet)
6. [Soal Domain Dosen](#-soal-domain-dosen-read-this)
7. [Hasil Model](#-hasil-model)

---

## 📁 Isi Proyek

```
prediksi-harga-rumah/
├── app.py             # Halaman web (tampilan, grafik, tombol)
├── model.py           # Data + machine learning (pembersihan, pelatihan model)
├── test_app.py        # Pengujian otomatis aplikasi
├── requirements.txt   # Daftar paket yang dibutuhkan
├── .streamlit/
│   └── config.toml    # Pengaturan tampilan & tema
└── .gitignore         # Daftar file yang tidak ikut di-upload
```

---

## 🚀 Cara Menjalankan

### Langkah 1 — Buka terminal di folder ini

Di Windows: klik kanan di folder `prediksi-harga-rumah` → **Open in Terminal**

### Langkah 2 — Aktifkan virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

Kalau muncul error "tidak dapat menjalankan skrip", jalankan ini sekali saja:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Langkah 3 — Jalankan aplikasinya

```powershell
streamlit run app.py
```

Browser akan terbuka otomatis di `http://localhost:8501`

Untuk menutup aplikasinya: tekan **Ctrl+C** di terminal.

---

## 📖 Penjelasan File

### `model.py` — Otak dari data dan machine learning

File ini tidak bisa dilihat hasilnya di browser, tapi isinya adalah
pekerjaan utama:

| Fungsi | Gunanya |
|---|---|
| `muat_data()` | Mengambil dataset dari scikit-learn |
| `bersihkan_data()` | Membuang data tidak valid + mendeteksi outlier |
| `latih_model()` | Melatih 2 model machine learning |
| `nilai_akurasi()` | Menghitung seberapa akurat model |
| `importance_fitur()` | Menilstik fitur mana yang paling berpengaruh |

### `app.py` — Tampilan web

| Elemen Streamlit | Dipakai untuk |
|---|---|
| `st.title`, `st.subheader` | Judul dan bagian |
| `st.slider` | Input angka yang bisa digeser |
| `st.metric` | Angka penting |
| `st.tabs` | Tab menu |
| `st.form` | Mengelompokkan input + tombol agar tidak langsung diproses |
| `st.plotly_chart` | Grafik interaktif |
| `st.cache_data` | Menarik dataset **satu kali saja** |

**Penting:** tanpa `@st.cache_data` dan `@st.cache_resource`, model akan
dilatih ulang setiap kali kamu menggeser slider. Itu akan sangat lambat.

---

## 🧠 Cara Kerja Aplikasinya

```
Dataset mentah (20.640 baris)
        ↓
   Pembersihan data
   (aturan fisik + metode IQR)
        ↓
  19.848 baris tersisa
        ↓
  Dibagi 80% latih : 20% uji
        ↓
  Latih 2 model
  (Linear Regression & Random Forest)
        ↓
   Ukur akurasi di data uji
        ↓
   Tampilkan ke pengguna
```

### Kenapa data dibagi 80:20?

Supaya akurasi model bisa diuji dengan **jujur**. Kalau model diuji
pakai data yang sama dengan data latih, hasilnya pasti kelihatan bagus
tapi tidak mencerminkan kemampuan model terhadap data baru.

### Kenapa pakai dua model?

Supaya ada pembanding. Dosen bisa melihat bahwa model yang lebih
kompleks (Random Forest) ternyata lebih akurat di sini.

---

## 🌐 Cara Deploy ke Internet

Pakai **Streamlit Community Cloud** — gratis, dan hasilnya bisa dibuka
siapa saja tanpa install apa pun.

### Langkah 1 — Unggah kode ke GitHub

```powershell
git init
git add .
git commit -m "Prediksi harga rumah"
git branch -M main
git remote add origin https://github.com/USERNAME-KAMU/prediksi-harga-rumah.git
git push -u origin main
```

> Ganti `USERNAME-KAMU` dengan username GitHub kamu.
> Repo **harus** publik (Public), kalau private gratis tidak bisa dipakai.

### Langkah 2 — Deploy

1. Buka <https://share.streamlit.io>
2. Masuk dengan akun GitHub
3. Klik **New app**
4. Pilih repository `prediksi-harga-rumah`
5. Branch: `main`
6. Biarkan yang lain default
7. Klik **Deploy**

Tunggu sekitar 2 menit, lalu kamu dapat alamat seperti:

```
https://namakamu-prediksi-harga-rumah.streamlit.app
```

### Catatan penting

- **Jangan** ikut mengunggah folder `.venv` — sudah otomatis diabaikan
  oleh `.gitignore`, dan ukurannya sangat besar
- File `requirements.txt` **harus** ada, karena Streamlit membacanya
  untuk tahu paket apa yang perlu dipasang
- Tiap kali kamu mengubah kode dan `git push`, aplikasinya otomatis
  diperbarui

---

## 🎓 Soal Domain Dosen (read this)

**Ini bagian yang perlu kamu konfirmasi ke dosen.**

Pengguna gratis Streamlit Community Cloud hanya mendapat alamat
`.streamlit.app` — **bukan** domain kustom seperti
`namakampus.ac.id/tugas-ml`.

Jadi ada beberapa kemungkinan:

| Kemungkinan | Yang perlu kamu lakukan |
|---|---|
| Dosen cukup ganti **alamat** web-nya ke `.streamlit.app` | Kirim URL-mu ke dosen, selesai |
| Dosen punya **subdomain** untuk tiap mahasiswa | Tanyakan subdomain apa yang harus dipakai, lalu dns diarahkan oleh dosen |
| Dosen mau **domain kustom** | Butuh paket berbayar Streamlit, atau di-proxy lewat layanan milik kampus |

**Saran:** tanyakan ke dosen sesuatu seperti:

> "Pak/Bu, untuk tahap deployment, apakah cukup saya kirim link
> `.streamlit.app`, atau ada subdomain/program studi yang harus
> saya pakai? Kalau harus diarahkan ke domain sendiri, kira-kira
> caranya bagaimana?"

Tanyakan **sebelum** kamu menyelesaikan pengerjaan, karena ini bisa
menambah langkah teknis yang di luar kendalimu.

---

## 📊 Hasil Model

Diukur pada 20% data uji yang belum pernah dilihat model:

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0,599 | $56.499 | $75.846 |
| **Random Forest** | **0,684** | **$48.702** | **$67.306** |

**Kesimpulan:** Random Forest lebih baik. Nilainya R² = 0,684 artinya
model ini bisa menjelaskan sekitar 68% variasi harga rumah. Fitur yang
paling berpengaruh adalah **pendapatan daerah**.

---

## 🧪 Menguji Aplikasi

```powershell
python test_app.py
```

Script ini menjalankan aplikasi tanpa browser dan memeriksa apakah
ada error, apakah semua tab tersedia, dan apakah tombol prediksi
berfungsi.

---

## 📚 Referensi

- **Sumber data**: California Housing (scikit-learn)
- **Dokumentasi Streamlit**: <https://docs.streamlit.io>
- **Dokumentasi scikit-learn**: <https://scikit-learn.org>
