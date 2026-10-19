3D HEART MODEL V1
=================
Penyelidik : Assoc Prof Ts Dr Rohayanti Hassan
Email      : rohayanti@utm.my
UTM-Claude Research Project

Projek ini boleh diletakkan di mana-mana folder. Semua arahan di bawah
ditaip dalam PowerShell/CMD dari folder projek (folder yang mengandungi
README ini).

SEDIAKAN PYTHON (sekali sahaja)
===============================
Python 3.9 disyorkan. Dengan uv:
  uv venv .venv --python 3.9
  uv pip install --python .venv\Scripts\python.exe -r requirements_3d.lock.txt
Tanpa uv (Python 3.9 sudah dipasang):
  python -m venv .venv
  .venv\Scripts\pip install -r requirements_3d.lock.txt
Laptop ARM (cth. Snapdragon): guna --python cpython-3.9-windows-x86_64-none
supaya SimpleITK dan scikit-image dapat dipasang.

CARA BUKA APP (DICOM VIEWER + 3D HEART MODEL)
=============================================
  .venv\Scripts\streamlit run scripts\heart_dicom_viewer.py

Kemudian buka browser: http://localhost:8501
Tab "DICOM Multi-View": masukkan path folder dataset, contoh
  <folder projek>\dataset
dan pilih siri CT (cth. CT CCTA DL) dari senarai.

STRUKTUR FOLDER
===============
  dataset\        <- Semua DICOM files (tidak dimuat naik ke server)
  scripts\        <- Python scripts
  output\         <- 3D models & visualizations
  tests\          <- Ujian regresi pipeline
  deploy\         <- Fail setup server (nginx, systemd)

3D HEART MODEL (Objektif 1)
===========================
  .venv\Scripts\python scripts\segment_heart.py
  .venv\Scripts\python scripts\eat_patient.py      (lemak epikardium, pilihan)
  .venv\Scripts\python scripts\calcium_score.py    (skor kalsium Agatston, pilihan)
  .venv\Scripts\python scripts\build_heart_model.py
Atau tekan butang "Run pipeline" dalam tab "3D Heart Model".
Butiran penuh: README_3D.md

LEMAK EPIKARDIUM (EAT)
======================
Diukur pada siri CT CALCIUM SCORE dengan model yang dilatih pada
dataset Rodriguez (Dice 0.81, 18 pesakit). Pesakit TC_PTID: 73.4 mL.
Ujian semula penanda aras (~50 minit):
  .venv\Scripts\python scripts\eat_benchmark.py --save-model

SKOR KALSIUM KORONARI (AGATSTON)
================================
Separa automatik: aplikasi mencadangkan lesi, pembaca mengesahkan
dalam tab "3D Heart Model" (bahagian Coronary calcium).
Pesakit TC_PTID: laporan radiologi = 0 Agatston.

UJIAN (TESTS)
=============
  uv pip install --python .venv\Scripts\python.exe -r requirements_dev.txt
  .venv\Scripts\python -m pytest -v
