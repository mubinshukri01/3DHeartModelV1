"""
==============================================================
  3D Heart Model V1 - ONE-CLICK SETUP
  UTM-Claude Research Project | rohayanti@utm.my

  Cara guna:
    1. Buka CMD atau PowerShell
    2. python run_everything.py
==============================================================
"""

import os, sys, subprocess

# project root = folder above scripts/, wherever it was unzipped (same rule as heart_io.py,
# which can't be imported here because SimpleITK may not be installed yet)
PROJECT = os.environ.get("HEART3D_ROOT",
                         os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.join(PROJECT, "dataset")
SCRIPTS = os.path.join(PROJECT, "scripts")
OUTPUT  = os.path.join(PROJECT, "output")
GDRIVE_FOLDER_ID = "1cDeInuzSeFwxlrJ9NhDSCM-su5tihKeR"
CALCIUM_FOLDER_ID = "13d0oJFsCXUoFx8sj46Vf60CaGAffZcf7"

def step(n, msg):
    print(f"\n{'─'*50}")
    print(f"  [{n}] {msg}")
    print(f"{'─'*50}")

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout: print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"  ⚠ {result.stderr[:300]}")
    return result.returncode == 0

print("\n" + "="*55)
print("  🫀  3D Heart Model V1 — Auto Setup & Download")
print("  UTM | rohayanti@utm.my")
print("="*55)

# ── STEP 1: Create folders ────────────────────────────────
step(1, "Mencipta folder projek")
for d in [DATASET, SCRIPTS, OUTPUT,
          os.path.join(OUTPUT, "3d_models"),
          os.path.join(OUTPUT, "visualizations")]:
    os.makedirs(d, exist_ok=True)
    print(f"  ✅ {d}")

# ── STEP 2: Install packages ──────────────────────────────
step(2, "Memasang Python packages")
# exact tested versions for the app + pipeline, plus gdown for the download step
lock = os.path.join(PROJECT, "requirements_3d.lock.txt")
for label, args in [("requirements_3d.lock.txt", f'-r "{lock}"'), ("gdown", "gdown")]:
    print(f"  Installing {label}...", end=" ", flush=True)
    ok = run(f'"{sys.executable}" -m pip install {args} --quiet')
    print("✅" if ok else "❌")

# ── STEP 3: Download dataset ──────────────────────────────
step(3, "Memuat turun dataset DICOM dari Google Drive")

try:
    import gdown
except:
    run(f'"{sys.executable}" -m pip install gdown --quiet')
    import gdown

print(f"\n  Folder ID : {GDRIVE_FOLDER_ID}")
print(f"  Destinasi : {DATASET}")
print(f"\n  ⏳ Sedang memuat turun... (mungkin 5-30 minit)\n")

# Cuba download keseluruhan folder dulu
try:
    gdown.download_folder(
        id=GDRIVE_FOLDER_ID,
        output=DATASET,
        quiet=False,
        use_cookies=False,
        remaining_ok=True
    )
    print("\n  ✅ Download folder utama selesai!")
except Exception as e:
    print(f"\n  ⚠ Folder download gagal ({e})")
    print("  Cuba download CT CALCIUM SCORE sahaja...")
    try:
        calcium_dir = os.path.join(DATASET, "CT_CALCIUM_SCORE")
        os.makedirs(calcium_dir, exist_ok=True)
        gdown.download_folder(
            id=CALCIUM_FOLDER_ID,
            output=calcium_dir,
            quiet=False,
            use_cookies=False,
            remaining_ok=True
        )
        print("\n  ✅ CT CALCIUM SCORE berjaya didownload!")
    except Exception as e2:
        print(f"\n  ❌ Download gagal: {e2}")
        print("""
  💡 Download manual:
     1. Buka: https://drive.google.com/drive/folders/1cDeInuzSeFwxlrJ9NhDSCM-su5tihKeR
     2. Klik kanan folder → Download
     3. Extract ke: """ + DATASET)

# ── STEP 4: Verify + count ────────────────────────────────
step(4, "Semak hasil download")
total, size = 0, 0
for root, _, files in os.walk(DATASET):
    for f in files:
        if f.endswith('.dcm'):
            total += 1
            size  += os.path.getsize(os.path.join(root, f))
print(f"  📁 Fail DICOM  : {total}")
print(f"  💾 Jumlah saiz : {size/1024/1024:.1f} MB")

# ── STEP 5: Write README (only if missing, so the real one is never overwritten)
readme = os.path.join(PROJECT, "README.txt")
if os.path.exists(readme):
    print("\n  ✅ README sudah ada, tidak ditulis semula")
else:
    with open(readme, "w", encoding="utf-8") as f:
        f.write(f"""3D HEART MODEL V1
=================
Penyelidik : Assoc Prof Ts Dr Rohayanti Hassan
Email      : rohayanti@utm.my
UTM-Claude Research Project

CARA BUKA DICOM VIEWER
======================
Buka CMD/PowerShell dan taip:
  cd {SCRIPTS}
  streamlit run heart_dicom_viewer.py

Kemudian buka browser: http://localhost:8501
Masukkan path dataset: {DATASET}
dan pilih siri CT dari senarai.

STRUKTUR FOLDER
===============
{PROJECT}
  dataset\\        <- Semua DICOM files
  scripts\\        <- Python scripts
  output\\         <- 3D models & visualizations
""")
    print("\n  ✅ README ditulis")

# ── DONE ─────────────────────────────────────────────────
print(f"""
{'='*55}
  ✅ SETUP SELESAI!
{'='*55}

  📁 Folder projek : {PROJECT}

  🚀 BUKA DICOM VIEWER:
     Buka CMD, kemudian:

     cd {SCRIPTS}
     streamlit run heart_dicom_viewer.py

     Browser akan buka: http://localhost:8501

{'='*55}
""")
input("Tekan Enter untuk keluar...")
