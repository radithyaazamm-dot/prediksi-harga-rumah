"""
test_app.py
Uji aplikasi tanpa membuka browser, memakai AppTest
(framework pengujian bawaan Streamlit).

Jalankan:
    python test_app.py
"""

from streamlit.testing.v1 import AppTest

print("=" * 60)
print("PENGUJIAN APLIKASI")
print("=" * 60)

# app.run() = jalankan aplikasi seperti kalau dibuka di browser
at = AppTest.from_file("app.py", default_timeout=600)
at.run()

# ---- Cek 1: aplikasi jalan tanpa error ----
if at.exception:
    print("[GAGAL] Ada error saat aplikasi dijalankan:")
    for e in at.exception:
        print("   ", e.value)
    raise SystemExit(1)

print("[OK] Aplikasi jalan tanpa error")

# ---- Cek 2: elemen penting tersedia ----
print(f"     - Judul      : {at.title[0].value}")
print(f"     - Tab        : {[t.label for t in at.tabs]}")
print(f"     - Sidebar    : {len(at.sidebar.slider)} slider, "
      f"{len(at.sidebar.subheader)} subheader")

assert len(at.tabs) == 3, "Harus ada 3 tab"
print("[OK] Semua 3 tab tersedia")

# ---- Cek 3: prediksi jalan saat form dikirim ----
print()
print("Menguji tombol prediksi...")

at2 = AppTest.from_file("app.py", default_timeout=600)
at2.run()

tombol = [b for b in at2.button if "Prediksi" in b.label]
print(f"     tombol ditemukan: {[b.label for b in tombol]}")

assert tombol, "Tombol prediksi tidak ditemukan"
tombol[0].click().run()

if at2.exception:
    print("[GAGAL] Error setelah tombol ditekan:")
    for e in at2.exception:
        print("   ", e.value)
    raise SystemExit(1)

print("[OK] Tombol prediksi ditekan tanpa error")

# ---- Cek 4: hasil prediksi muncul ----
teks = " ".join(e.value for e in at2.markdown if hasattr(e, "value"))
print(f"     judul hasil: "
      f"{'ada' if 'Hasil Prediksi' in teks else 'TIDAK ADA'}")

assert "Hasil Prediksi" in teks, "Hasil prediksi tidak muncul"
print("[OK] Hasil prediksi berhasil ditampilkan")

print()
print("=" * 60)
print("SEMUA PENGUJIAN LULUS")
print("=" * 60)
