import streamlit as st
import pandas as pd
import joblib
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="Early Warning System Dropout Siswa | Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide"
)

# Load Model Pipeline
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "student_dropout_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}. Pastikan model/student_dropout_model.pkl tersedia.")
    st.stop()

# Header Dashboard
st.title("🎓 Jaya Jaya Institut - Dropout Early Warning System")
st.markdown("""
Sistem deteksi dini performa akademik mahasiswa untuk mengidentifikasi indikasi putus kuliah (*dropout*) 
sejak dini sehingga pihak kampus dapat segera memberikan pendampingan akademik dan konseling finansial.
""")

st.sidebar.header("Pilihan Mode Prediksi")
mode = st.sidebar.radio("Metode Input:", ["Prediksi Individual", "Batch Prediksi (Upload CSV)"])

if mode == "Prediksi Individual":
    st.subheader("📋 Form Data Mahasiswa")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Usia Saat Masuk (Age at enrollment)", min_value=16, max_value=70, value=20)
        gender = st.selectbox("Jenis Kelamin", options=[1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
        scholarship = st.selectbox("Penerima Beasiswa", options=[1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        debtor = st.selectbox("Memiliki Tunggakan/Hutang", options=[0, 1], format_func=lambda x: "Ada Tunggakan" if x == 1 else "Lunas / Bebas Hutang")
        tuition_fees = st.selectbox("SPP Terbayar Tepat Waktu", options=[1, 0], format_func=lambda x: "Ya (Lunas)" if x == 1 else "Tidak (Menunggak)")

    with col2:
        sem1_enrolled = st.number_input("SKS Diambil Semester 1", min_value=0, max_value=30, value=6)
        sem1_approved = st.number_input("SKS Lulus Semester 1", min_value=0, max_value=30, value=5)
        sem1_grade = st.number_input("Rata-rata Nilai Semester 1 (Skala 0-20)", min_value=0.0, max_value=20.0, value=12.5)
        admission_grade = st.number_input("Nilai Seleksi Masuk (Admission Grade)", min_value=0.0, max_value=200.0, value=125.0)

    with col3:
        sem2_enrolled = st.number_input("SKS Diambil Semester 2", min_value=0, max_value=30, value=6)
        sem2_approved = st.number_input("SKS Lulus Semester 2", min_value=0, max_value=30, value=4)
        sem2_grade = st.number_input("Rata-rata Nilai Semester 2 (Skala 0-20)", min_value=0.0, max_value=20.0, value=11.5)

    if st.button("Jalankan Prediksi Risiko", type="primary"):
        # Format input disesuaikan dengan skema fitur dataset asli
        input_dict = {
            'Marital_status': [1],
            'Application_mode': [1],
            'Application_order': [1],
            'Course': [9500],
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
        
        input_df = pd.DataFrame(input_dict)
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.divider()
        if pred == 1:
            st.error(f"🚨 **Peringatan Risiko Tinggi: Mahasiswa Berpotensi Dropout!**")
            st.metric("Probabilitas Dropout", f"{prob * 100:.2f}%")
            st.warning("Rekomendasi Tindakan Segera: Jadwalkan bimbingan konseling akademik dan pemeriksaan kendala keuangan.")
        else:
            st.success(f"✅ **Risiko Rendah: Mahasiswa Cenderung Menyelesaikan Studi.**")
            st.metric("Probabilitas Aman", f"{(1 - prob) * 100:.2f}%")

else:
    st.subheader("📂 Batch Prediksi via CSV")
    uploaded_file = st.file_uploader("Unggah file CSV data mahasiswa", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file, sep=';')
            if batch_df.shape[1] == 1:
                uploaded_file.seek(0)
                batch_df = pd.read_csv(uploaded_file, sep=',')
        except:
            uploaded_file.seek(0)
            batch_df = pd.read_csv(uploaded_file)

        st.write("Data Berhasil Dimuat:", batch_df.head())
        if st.button("Proses Seluruh Data"):
            clean_test = batch_df.drop(columns=['Status'], errors='ignore')
            preds = model.predict(clean_test)
            probs = model.predict_proba(clean_test)[:, 1]

            batch_df['Prediksi_Dropout'] = ["Dropout" if p == 1 else "Aman" for p in preds]
            batch_df['Probabilitas_Dropout'] = [f"{p*100:.2f}%" for p in probs]

            st.dataframe(batch_df[['Prediksi_Dropout', 'Probabilitas_Dropout'] + [c for c in batch_df.columns if c not in ['Prediksi_Dropout', 'Probabilitas_Dropout']]])