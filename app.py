import os
import joblib
import pandas as pd
import streamlit as st

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Dropout Early Warning System - Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS agar Tombol Menjadi Berwarna Mencolok
st.markdown("""
<style>
/* Styling Tombol Form Submit Primary */
div.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(30, 60, 114, 0.35) !important;
    transition: all 0.3s ease !important;
}

div.stButton > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(30, 60, 114, 0.45) !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# Fungsi Memuat Model Machine Learning
@st.cache_resource
def load_trained_model():
    model_path = os.path.join("model", "student_dropout_model.pkl")
    if not os.path.exists(model_path):
        model_path = "student_dropout_model.pkl"
    return joblib.load(model_path)

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.info("Pastikan berkas model 'student_dropout_model.pkl' sudah disimpan di dalam folder 'model/'.")
    st.stop()

# Header Aplikasi
st.title("🎓 Jaya Jaya Institut - Dropout Early Warning System")
st.markdown("""
Aplikasi sistem deteksi dini untuk memprediksi potensi keberhasilan studi mahasiswa (**Dropout** vs **Graduate**). 
Sistem ini memanfaatkan algoritma Machine Learning terlatih untuk membantu dosen pembimbing akademik dan pihak manajemen 
dalam memberikan intervensi tepat waktu sebelum mahasiswa memutuskan berhenti kuliah.
""")

st.sidebar.header("Pilihan Navigasi")
menu = st.sidebar.radio(
    "Pilih Mode Evaluasi:",
    ["Prediksi Individual (Formulir)", "Prediksi Massal (Unggah Berkas CSV)"]
)

# Mapping Jurusan untuk Input Formulir
COURSE_OPTIONS = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)"
}

# MODE 1: PREDIKSI INDIVIDUAL
if menu == "Prediksi Individual (Formulir)":
    st.subheader("📋 Input Data Indikator Mahasiswa")
    st.caption("Masukkan data profil demografi, status finansial, dan capaian akademik semester awal mahasiswa.")

    with st.form("individual_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Profil Mahasiswa & Finansial**")
            course_id = st.selectbox(
                "Program Studi / Jurusan:",
                options=list(COURSE_OPTIONS.keys()),
                format_func=lambda x: f"{x} - {COURSE_OPTIONS[x]}"
            )
            age = st.number_input("Usia Saat Masuk Kuliah:", min_value=15, max_value=70, value=20)
            gender = st.selectbox("Jenis Kelamin:", options=[1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
            tuition_fees = st.selectbox("Status SPP:", options=[1, 0], format_func=lambda x: "Lunas / Tepat Waktu" if x == 1 else "Menunggak")
            debtor = st.selectbox("Status Hutang / Pinjaman:", options=[0, 1], format_func=lambda x: "Bebas Tunggakan" if x == 0 else "Ada Tunggakan")
            scholarship = st.selectbox("Penerima Beasiswa:", options=[1, 0], format_func=lambda x: "Ya (Penerima)" if x == 1 else "Tidak")

        with col2:
            st.markdown("**Performa Akademik Semester 1**")
            admission_grade = st.number_input("Nilai Seleksi Masuk (0 - 200):", min_value=0.0, max_value=200.0, value=125.0)
            sem1_enrolled = st.number_input("SKS Diambil Semester 1:", min_value=0, max_value=30, value=6)
            sem1_approved = st.number_input("SKS Lulus Semester 1:", min_value=0, max_value=30, value=5)
            sem1_grade = st.number_input("Rata-rata Nilai Semester 1 (Skala 0 - 20):", min_value=0.0, max_value=20.0, value=12.5)

        with col3:
            st.markdown("**Performa Akademik Semester 2**")
            sem2_enrolled = st.number_input("SKS Diambil Semester 2:", min_value=0, max_value=30, value=6)
            sem2_approved = st.number_input("SKS Lulus Semester 2:", min_value=0, max_value=30, value=4)
            sem2_grade = st.number_input("Rata-rata Nilai Semester 2 (Skala 0 - 20):", min_value=0.0, max_value=20.0, value=11.5)

        submit_button = st.form_submit_button("🔍 Analisis Risiko Mahasiswa", type="primary", use_container_width=True)

    if submit_button:
        input_data = {
            'Marital_status': [1],
            'Application_mode': [1],
            'Application_order': [1],
            'Course': [course_id],
            'Daytime_evening_attendance': [1],
            'Previous_qualification': [1],
            'Previous_qualification_grade': [120.0],
            'Nacionality': [1],
            'Mothers_qualification': [1],
            'Fathers_qualification': [1],
            'Mothers_occupation': [5],
            'Fathers_occupation': [5],
            'Admission_grade': [admission_grade],
            'Displaced': [1],
            'Educational_special_needs': [0],
            'Debtor': [debtor],
            'Tuition_fees_up_to_date': [tuition_fees],
            'Gender': [gender],
            'Scholarship_holder': [scholarship],
            'Age_at_enrollment': [age],
            'International': [0],
            'Curricular_units_1st_sem_credited': [0],
            'Curricular_units_1st_sem_enrolled': [sem1_enrolled],
            'Curricular_units_1st_sem_evaluations': [sem1_enrolled],
            'Curricular_units_1st_sem_approved': [sem1_approved],
            'Curricular_units_1st_sem_grade': [sem1_grade],
            'Curricular_units_1st_sem_without_evaluations': [0],
            'Curricular_units_2nd_sem_credited': [0],
            'Curricular_units_2nd_sem_enrolled': [sem2_enrolled],
            'Curricular_units_2nd_sem_evaluations': [sem2_enrolled],
            'Curricular_units_2nd_sem_approved': [sem2_approved],
            'Curricular_units_2nd_sem_grade': [sem2_grade],
            'Curricular_units_2nd_sem_without_evaluations': [0],
            'Unemployment_rate': [10.8],
            'Inflation_rate': [1.4],
            'GDP': [1.74]
        }

        input_df = pd.DataFrame(input_data)
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        dropout_proba = probabilities[1]

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if prediction == 1:
                st.error("### 🚨 Potensi Risiko: DROPOUT")
                st.metric("Tingkat Risiko Dropout", f"{dropout_proba * 100:.2f}%")
            else:
                st.success("### ✅ Potensi Keberhasilan: GRADUATE")
                st.metric("Peluang Kelulusan", f"{(1 - dropout_proba) * 100:.2f}%")

        with res_col2:
            st.markdown("**Rekomendasi Tindakan Cepat:**")
            if prediction == 1:
                st.warning("""
                - **Intervensi Finansial Segera:** Lakukan konfirmasi apakah mahasiswa mengalami kendala SPP atau tunggakan hutang. Tawarkan skema cicilan atau beasiswa darurat.
                - **Pendampingan Akademik:** Jadwalkan konsultasi khusus bersama Dosen Wali untuk mengevaluasi rendahnya perolehan SKS di semester awal.
                """)
            else:
                st.info("""
                - Mahasiswa berada dalam jalur akademik yang sehat.
                - Pertahankan performa dengan monitoring berkala di setiap akhir semester.
                """)

# MODE 2: BATCH PREDICTION
else:
    st.subheader("📂 Batch Prediksi Mahasiswa Aktif (Enrolled)")
    st.markdown("""
    Unggah berkas CSV yang berisi data mahasiswa (seperti data `unseen_enrolled_students.csv` yang telah dipisahkan saat tahap Data Preparation) 
    untuk memproses evaluasi risiko seluruh mahasiswa aktif secara bersamaan.
    """)

    uploaded_file = st.file_uploader("Unggah File Data CSV:", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file, sep=';')
            if batch_df.shape[1] == 1:
                uploaded_file.seek(0)
                batch_df = pd.read_csv(uploaded_file, sep=',')
        except Exception:
            uploaded_file.seek(0)
            batch_df = pd.read_csv(uploaded_file)

        st.write(f"**Data Berhasil Dimuat:** {batch_df.shape[0]} baris ditemukan.")
        st.dataframe(batch_df.head())

        if st.button("🚀 Jalankan Prediksi Seluruh Data", type="primary"):
            features_to_predict = batch_df.drop(columns=['Status', 'Target', 'is_dropout'], errors='ignore')

            predictions = model.predict(features_to_predict)
            probabilities = model.predict_proba(features_to_predict)[:, 1]

            results = batch_df.copy()
            results['Prediksi Status'] = ["Dropout" if p == 1 else "Graduate" for p in predictions]
            results['Probabilitas Dropout (%)'] = [round(p * 100, 2) for p in probabilities]

            st.divider()
            st.subheader("📊 Hasil Evaluasi Mahasiswa")

            kpi1, kpi2, kpi3 = st.columns(3)
            total_students = len(results)
            dropout_count = sum(predictions == 1)
            graduate_count = sum(predictions == 0)

            kpi1.metric("Total Mahasiswa Dievaluasi", total_students)
            kpi2.metric("Berpotensi Dropout", dropout_count, delta=f"{(dropout_count/total_students)*100:.1f}%", delta_color="inverse")
            kpi3.metric("Berpotensi Lulus", graduate_count)

            output_cols = ['Prediksi Status', 'Probabilitas Dropout (%)'] + [c for c in batch_df.columns if c not in ['Prediksi Status', 'Probabilitas Dropout (%)']]
            st.dataframe(results[output_cols])

            csv_export = results[output_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Hasil Prediksi (CSV)",
                data=csv_export,
                file_name="hasil_prediksi_mahasiswa.csv",
                mime="text/csv"
            )