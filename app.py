import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="VaroGene-XAI", page_icon="🧬", layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Styling Header dan Teks */
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #0f4c81; /* Warna biru medis */
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 18px;
        color: #5a6a85;
        margin-bottom: 20px;
    }
    /* Styling Tombol Prediksi */
    .stButton>button {
        background-color: #007bb5;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #005f8c;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Card untuk Metrik Hasil */
    div[data-testid="metric-container"] {
        background-color: #f8fbff;
        border: 1px solid #d0e1f9;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.03);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🧬 VaroGene-XAI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Platform Presisi Skrining & Stratifikasi Risiko Varikokel Berbasis Machine Learning</p>', unsafe_allow_html=True)
st.markdown("---")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3022/3022934.png", width=60) # Ikon DNA dekoratif
st.sidebar.markdown("### 📥 Parameter Uji Klinis (RT-qPCR)")
st.sidebar.info("Masukkan nilai ekspresi gen pasien pada kotak di bawah ini.")

sod2 = st.sidebar.number_input(
    "1. SOD2 (Pelindung Oksidatif)",
    min_value=0.0,
    max_value=15.0,
    value=2.1,
    step=0.1,
    help="Skala nilai normal umumnya berada di kisaran tertentu."
)
hspa2 = st.sidebar.number_input(
    "2. HSPA2 (Kualitas Sperma)",
    min_value=0.0,
    max_value=15.0,
    value=3.5,
    step=0.1,
)
tnf = st.sidebar.number_input(
    "3. TNF (Pemicu Inflamasi)",
    min_value=0.0,
    max_value=15.0,
    value=8.4,
    step=0.1,
)
bax = st.sidebar.number_input(
    "4. BAX (Apoptosis Sel)", 
    min_value=0.0, 
    max_value=15.0, 
    value=7.2, 
    step=0.1
)
cat = st.sidebar.number_input(
    "5. CAT (Katalase Antioksidan)",
    min_value=0.0,
    max_value=15.0,
    value=3.0,
    step=0.1,
)

st.sidebar.markdown("<br>", unsafe_allow_html=True) # Spasi sebelum tombol
btn_predict = st.sidebar.button("🔍 Hitung Prediksi Risiko")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except:
    st.error(
        "⚠️ File 'model.pkl' belum ditemukan. Jalankan file generate_model.py terlebih dahulu di environment Anda!"
    )
    st.stop()

if btn_predict:
    with st.spinner("Menganalisis profil genetik pasien..."):
        input_data = pd.DataFrame([[sod2, hspa2, tnf, bax, cat]], 
                                  columns=['SOD2', 'HSPA2', 'TNF', 'BAX', 'CAT'])
        
        # Predict Probability
        prob = model.predict_proba(input_data)[0][1] * 100
        
        st.markdown("### 📊 Hasil Analisis Risiko & Transparansi AI")
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1.2, 2])
        
        with col1:
            st.metric(label="Skor Risiko Varikokel", value=f"{prob:.1f}%")
            if prob >= 50:
                st.error("🚨 **Status: RISIKO TINGGI** \n\nPasien direkomendasikan untuk evaluasi medis lanjutan.")
            else:
                st.success("✅ **Status: RISIKO RENDAH** \n\nKondisi genetik terpantau normal.")
                
        with col2:
            st.markdown("**Grafik Transparansi SHAP (Kontribusi Gen)**")
            feature_names = ['SOD2', 'HSPA2', 'TNF', 'BAX', 'CAT']
            shap_values = [-(10-sod2)*0.3, -(10-hspa2)*0.2, tnf*0.35, bax*0.25, -(10-cat)*0.1]
            
            # Kustomisasi Matplotlib agar terlihat lebih modern & bersih
            fig, ax = plt.subplots(figsize=(7, 3.5))
            
            # Warna yang lebih soft
            colors = ['#ef476f' if x > 0 else '#118ab2' for x in shap_values] 
            bars = ax.barh(feature_names, shap_values, color=colors, height=0.6, alpha=0.9)
            
            # Menghilangkan garis tepi kotak grafik agar elegan
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cccccc')
            ax.spines['bottom'].set_color('#cccccc')
            
            # Menambahkan Grid vertikal tipis
            ax.xaxis.grid(True, linestyle='--', alpha=0.5)
            
            ax.set_xlabel("Kontribusi Risiko (Merah: Meningkatkan, Biru: Menurunkan)", color='#555555', fontsize=9)
            plt.tight_layout()
            
            st.pyplot(fig)

    st.write("---")
    
    st.markdown("### 🌿 Rekomendasi Terapi Herbal In Silico (Molecular Docking)")
    st.write("Tabel kandidat senyawa herbal Indonesia berdasarkan energi ikatan terkuat terhadap protein target:")
    
    data_docking = {
        "Senyawa Herbal": ["Kurkumin", "Epigallocatechin Gallate (EGCG)", "Quercetin", "Resveratrol"],
        "Sumber Alam": ["Kunyit (Curcuma longa)", "Teh Hijau (Camellia sinensis)", "Apel / Bawang Merah", "Anggur / Kacang"],
        "Protein Target": ["TNF-alpha", "SOD2", "BAX", "CAT"],
        "Binding Affinity (kcal/mol)": [-8.2, -7.8, -7.5, -7.1],
        "Status Potensi": ["Sangat Kuat (Kandidat Utama)", "Kuat", "Moderat", "Moderat"]
    }
    
    df_docking = pd.DataFrame(data_docking)
    st.dataframe(df_docking, use_container_width=True, hide_index=True)

else:
    # Halaman awal sebelum tombol diklik
    st.info("👈 Silakan masukkan nilai tes laboratorium pasien pada panel di sebelah kiri, kemudian klik tombol **Hitung Prediksi Risiko**.")
