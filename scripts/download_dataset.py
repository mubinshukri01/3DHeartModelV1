"""
==============================================================
  Download CT Cardiac Procedure Dataset dari Google Drive
  UTM-Claude Research Project | rohayanti@utm.my
==============================================================
  Cara guna:
    python download_dataset.py

  Dataset: TC_PTID / CT CARDIAC PROCEDURE
  Google Drive: https://drive.google.com/drive/folders/1cDeInuzSeFwxlrJ9NhDSCM-su5tihKeR
==============================================================
"""

import os
import sys
import time

# ─── Check & install gdown ────────────────────────────────
try:
    import gdown
except ImportError:
    print("Memasang gdown...")
    os.system(f"{sys.executable} -m pip install gdown --quiet")
    import gdown

try:
    from tqdm import tqdm
except ImportError:
    os.system(f"{sys.executable} -m pip install tqdm --quiet")

# ─── CONFIG ───────────────────────────────────────────────
# project root = folder above scripts/, wherever it was unzipped (same rule as heart_io.py)
PROJECT_DIR  = os.environ.get("HEART3D_ROOT",
                              os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR  = os.path.join(PROJECT_DIR, "dataset")

# Google Drive Folder IDs (dari eksplorasi folder)
MAIN_FOLDER_ID = "1cDeInuzSeFwxlrJ9NhDSCM-su5tihKeR"

# Sub-folder IDs (dari CT CARDIAC PROCEDURE)
SERIES_FOLDERS = {
    "CT_CALCIUM_SCORE":                 "13d0oJFsCXUoFx8sj46Vf60CaGAffZcf7",
    "CT_OCTA_DL":                       None,  # akan download dari main
    "CT_LUNGS_0625MM":                  None,
    "CT_MEDIASTINUM_DL":                None,
    "CT_PatientReport":                 None,
    "CT_Scout":                         None,
    "CT_SSFreeze_116_Original_302":     None,
    "SR_Dose_Record":                   None,
    "SR_FUJI_HL7_Radiological_Report":  None,
}

# ─── FUNCTIONS ────────────────────────────────────────────
def print_banner():
    print("\n" + "="*60)
    print("  🫀  CT CARDIAC PROCEDURE - Dataset Downloader")
    print("  UTM-Claude Research Project")
    print("  rohayanti@utm.my")
    print("="*60 + "\n")


def create_folders():
    """Buat struktur folder projek."""
    folders = [
        PROJECT_DIR,
        DATASET_DIR,
        os.path.join(PROJECT_DIR, "scripts"),
        os.path.join(PROJECT_DIR, "output"),
        os.path.join(PROJECT_DIR, "output", "3d_models"),
        os.path.join(PROJECT_DIR, "output", "visualizations"),
    ]
    for f in folders:
        os.makedirs(f, exist_ok=True)
    print(f"✅ Struktur folder dicipta di: {PROJECT_DIR}\n")


def download_main_folder():
    """Download keseluruhan folder CT CARDIAC PROCEDURE."""
    print("📥 Memuat turun dataset dari Google Drive...")
    print(f"   Folder: CT CARDIAC PROCEDURE")
    print(f"   Destinasi: {DATASET_DIR}")
    print(f"   (Ini mungkin mengambil masa 5-30 minit bergantung kepada saiz)\n")

    url = f"https://drive.google.com/drive/folders/{MAIN_FOLDER_ID}"

    try:
        gdown.download_folder(
            url=url,
            output=DATASET_DIR,
            quiet=False,
            use_cookies=False,
            remaining_ok=True
        )
        print("\n✅ Download selesai!")
        return True
    except Exception as e:
        print(f"\n⚠️  Ralat semasa download: {e}")
        print("\nCuba kaedah alternatif...")
        return download_individual_series()


def download_individual_series():
    """Download setiap series secara berasingan jika folder download gagal."""
    print("\n📥 Cuba download siri CT satu persatu...")

    success_count = 0

    # Download CT CALCIUM SCORE (kita tahu ID-nya)
    calcium_dir = os.path.join(DATASET_DIR, "CT_CALCIUM_SCORE")
    os.makedirs(calcium_dir, exist_ok=True)

    print(f"\n[1/9] CT CALCIUM SCORE...")
    try:
        url = f"https://drive.google.com/drive/folders/13d0oJFsCXUoFx8sj46Vf60CaGAffZcf7"
        gdown.download_folder(
            url=url,
            output=calcium_dir,
            quiet=False,
            use_cookies=False,
            remaining_ok=True
        )
        print(f"   ✅ CT CALCIUM SCORE - Selesai")
        success_count += 1
    except Exception as e:
        print(f"   ❌ Gagal: {e}")

    if success_count == 0:
        print("\n❌ Download gagal. Kemungkinan sebab:")
        print("   1. Sambungan internet tidak stabil")
        print("   2. Google Drive had download")
        print("   3. Perlu log masuk Google")
        print("\n💡 Alternatif: Download manual dari:")
        print(f"   https://drive.google.com/drive/folders/{MAIN_FOLDER_ID}")
        print(f"\n   Simpan ke: {DATASET_DIR}")
        return False

    return True


def verify_download():
    """Semak fail yang berjaya didownload."""
    print("\n🔍 Mengesahkan download...")
    total_files = 0
    total_size  = 0

    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if f.endswith('.dcm'):
                fpath = os.path.join(root, f)
                total_files += 1
                total_size  += os.path.getsize(fpath)

    size_mb = total_size / (1024 * 1024)
    print(f"   📁 Jumlah fail DICOM : {total_files}")
    print(f"   💾 Jumlah saiz       : {size_mb:.1f} MB")

    if total_files > 0:
        print(f"\n✅ Dataset berjaya dimuat turun!")
        return True
    else:
        print(f"\n⚠️  Tiada fail DICOM dijumpai dalam {DATASET_DIR}")
        return False


def create_readme():
    """Buat README untuk folder projek (hanya jika belum ada)."""
    readme_path = os.path.join(PROJECT_DIR, "README.txt")
    if os.path.exists(readme_path):
        print(f"\n📄 README sudah ada, tidak ditulis semula: {readme_path}")
        return
    content = f"""
3D HEART MODEL V1 - UTM Research Project
=========================================
Penyelidik  : Assoc Prof Ts Dr Rohayanti Hassan
Email       : rohayanti@utm.my
Institusi   : Universiti Teknologi Malaysia (UTM)

STRUKTUR FOLDER
===============
3DHeartModelV1/
├── dataset/                        <- Dataset DICOM
│   ├── CT CALCIUM SCORE/           <- 50 slices CAC scoring
│   ├── CT CCTA DL/                 <- 224 slices CCTA, Deep Learning recon
│   ├── CT LUNGS 0.625MM/           <- High-res lung CT
│   ├── CT MEDIASTINUM DL/          <- Mediastinal CT + DL
│   ├── CT PatientReport/           <- Laporan klinikal
│   ├── CT Scout/                   <- Scout/topogram images
│   ├── CT SSFreeze 116 Original Series 302/  <- Motion-corrected cardiac CT
│   ├── SR Dose Record/             <- DICOM dose SR
│   └── SR FUJI Basic Text SR for HL7 Radiological Report/
│
├── scripts/
│   ├── heart_dicom_viewer.py       <- DICOM Viewer + 3D Heart Model (Streamlit)
│   └── download_dataset.py         <- Script ini
│
└── output/
    ├── 3d_models/                  <- Output 3D mesh (.stl, .obj, .glb)
    └── visualizations/             <- Gambar QC

CARA GUNA DICOM VIEWER
======================
1. Buka CMD/PowerShell
2. cd {os.path.join(PROJECT_DIR, 'scripts')}
3. streamlit run heart_dicom_viewer.py
4. Buka browser: http://localhost:8501
5. Masukkan path: {DATASET_DIR}  dan pilih siri CT dari senarai

SUMBER DATASET
==============
Google Drive: https://drive.google.com/drive/folders/1cDeInuzSeFwxlrJ9NhDSCM-su5tihKeR
Owner       : utm4dt@gmail.com
Pesakit     : TC_PTID (1 pesakit, pelbagai siri CT)

OBJEKTIF PROJEK
===============
1. Optimasi rendering 3D model dinamik untuk animasi kardiovaskular
2. Pembangunan simulator extended reality (XR) untuk rawatan kardiovaskular
3. Algoritma AI untuk memahami dinamik sistem kardiovaskular
"""
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n📄 README dicipta: {readme_path}")


def print_next_steps():
    """Tunjuk langkah seterusnya."""
    print("\n" + "="*60)
    print("  ✅ SETUP SELESAI!")
    print("="*60)
    print(f"""
📁 Folder Projek  : {PROJECT_DIR}
📊 Dataset        : {DATASET_DIR}

🚀 LANGKAH SETERUSNYA:
────────────────────
1. Buka DICOM Viewer:
   cd {os.path.join(PROJECT_DIR, 'scripts')}
   streamlit run heart_dicom_viewer.py

2. Dalam app, masukkan path:
   {DATASET_DIR}

3. Pilih siri 'CT CALCIUM SCORE', window preset
   'Bone/CAC' dan scroll slices!

💡 Tip GPU: App ini jalan di CPU. Untuk 3D
   rendering, kita akan gunakan GPU pipeline
   seterusnya.
""")


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    print_banner()

    # 1. Buat folders
    create_folders()

    # 2. Download dataset
    print("Adakah anda mahu download dataset sekarang?")
    print("(Dataset mungkin besar - beberapa ratus MB)")
    choice = input("\nDownload sekarang? (y/n): ").strip().lower()

    if choice == 'y':
        success = download_main_folder()
        if success:
            verify_download()
    else:
        print("\n⏭️  Download dilangkau.")
        print(f"   Anda boleh download manual ke: {DATASET_DIR}")

    # 3. Buat README
    create_readme()

    # 4. Tunjuk langkah seterusnya
    print_next_steps()

    input("\nTekan Enter untuk keluar...")
