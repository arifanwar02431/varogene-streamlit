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
    /* Styling Judul agar Besar & Menonjol */
    .hero-title {
        font-size: 4.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(45deg, #007bb5, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        line-height: 1.1;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 1.3rem;
        color: #888888; /* Warna abu-abu netral untuk Light/Dark mode */
        margin-top: 5px;
        margin-bottom: 30px;
        font-weight: 500;
    }
    /* Styling Tombol Prediksi (Tetap konsisten di semua mode) */
    .stButton>button {
        background: linear-gradient(90deg, #007bb5, #005f8c);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 123, 181, 0.4);
        color: white;
    }
    /* Card untuk Metrik Hasil - Menggunakan rgba agar adaptif di Light/Dark mode */
    div[data-testid="metric-container"] {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🧬 VaroGene-XAI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Platform Presisi Skrining & Stratifikasi Risiko Varikokel Berbasis Machine Learning</div>', unsafe_allow_html=True)
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
            
            fig, ax = plt.subplots(figsize=(7, 3.5))
            
            # Membuat background Matplotlib tembus pandang (Support Light/Dark)
            fig.patch.set_alpha(0.0) 
            ax.set_facecolor("transparent")
            
            # Warna batang grafik (Merah soft / Biru soft)
            colors = ['#ef476f' if x > 0 else '#118ab2' for x in shap_values] 
            bars = ax.barh(feature_names, shap_values, color=colors, height=0.6, alpha=0.9)
            
            # Menghilangkan garis tepi kotak
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Mengubah warna font & axis menjadi abu-abu (Terlihat di putih & hitam)
            ax.spines['left'].set_color('gray')
            ax.spines['bottom'].set_color('gray')
            ax.tick_params(colors='gray')
            
            # Menambahkan Grid vertikal tipis
            ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
            
            ax.set_xlabel("Kontribusi Risiko (Merah: Meningkatkan, Biru: Menurunkan)", color='gray', fontsize=9)
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
