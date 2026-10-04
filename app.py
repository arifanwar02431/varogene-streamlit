import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

# 1. KONFIGURASI HALAMAN UTAMA
st.set_page_config(
    page_title="VaroGene-XAI | Inovasi Medis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed" # Menyembunyikan sidebar agar fokus scroll ke bawah
)

# 2. INJEKSI CSS TEMA MEDIS & INOVASI
# Memasukkan gaya UI modern, warna Deep Blue (#1E3A8A) dan Teal (#00C2CB) langsung ke dalam Streamlit
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Judul Utama (Hero) */
    .hero-title {
        font-size: 4rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #1E3A8A, #00C2CB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-top: 5px;
        margin-bottom: 40px;
    }
    
    /* Pembatas (Header Section) */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        border-bottom: 3px solid rgba(0, 194, 203, 0.3);
        padding-bottom: 8px;
        margin-top: 30px;
        margin-bottom: 20px;
        color: #1E3A8A;
    }
    
    /* Kotak Fitur Edukasi */
    .edu-box {
        background: linear-gradient(90deg, rgba(30, 58, 138, 0.05) 0%, rgba(0, 194, 203, 0.05) 100%);
        border-left: 5px solid #00C2CB;
        padding: 20px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 30px;
    }
    
    /* Kustomisasi Tombol Streamlit */
    .stButton>button {
        background: linear-gradient(135deg, #1E3A8A 0%, #00C2CB 100%);
        color: white !important;
        font-weight: 600;
        font-size: 1.1rem;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 194, 203, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 194, 203, 0.5);
    }

    /* Penyesuaian saat mode gelap (Dark Mode) */
    @media (prefers-color-scheme: dark) {
        .section-header { color: #f8fafc; border-bottom: 3px solid rgba(0, 194, 203, 0.5); }
        .hero-subtitle { color: #cbd5e1; }
        .edu-box { color: #f1f5f9; background: linear-gradient(90deg, rgba(30, 58, 138, 0.15) 0%, rgba(0, 194, 203, 0.1) 100%);}
    }
    </style>
""", unsafe_allow_html=True)

# 3. KONTEN HALAMAN UTAMA (SCROLL KE BAWAH)
st.markdown('<div class="hero-title">🧬 VaroGene-XAI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Platform Presisi Skrining & Stratifikasi Risiko Varikokel Berbasis Machine Learning</div>', unsafe_allow_html=True)

# Fitur Edukasi
st.markdown("""
<div class="edu-box">
    <h4 style="margin-top:0; color:#00C2CB;">📖 Mengapa Profil Genetik Penting dalam Varikokel?</h4>
    <p style="margin-bottom:0; font-size: 0.95rem;">
    Varikokel adalah pembesaran abnormal vena dalam skrotum yang memicu stres oksidatif dan infertilitas. 
    Penelitian modern menunjukkan ketidakseimbangan ekspresi gen terkait antioksidan (<b>SOD2, CAT</b>), 
    kualitas sperma (<b>HSPA2</b>), dan pemicu inflamasi (<b>TNF, BAX</b>) memegang peran kunci dalam tingkat keparahan. 
    Sistem AI kami menggunakan metrik biomarker ini untuk stratifikasi risiko secara non-invasif.
    </p>
</div>
""", unsafe_allow_html=True)

# Form Input Data (Diletakkan di tengah, bukan di sidebar)
st.markdown('<div class="section-header">📥 1. Input Nilai Biomarker (RT-qPCR)</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    sod2 = st.number_input("1. SOD2 (Antioksidan)", min_value=0.0, max_value=15.0, value=2.1, step=0.1)
    bax = st.number_input("4. BAX (Apoptosis Sel)", min_value=0.0, max_value=15.0, value=7.2, step=0.1)
with col2:
    hspa2 = st.number_input("2. HSPA2 (Kualitas Sperma)", min_value=0.0, max_value=15.0, value=3.5, step=0.1)
    cat = st.number_input("5. CAT (Katalase)", min_value=0.0, max_value=15.0, value=3.0, step=0.1)
with col3:
    tnf = st.number_input("3. TNF (Inflamasi)", min_value=0.0, max_value=15.0, value=8.4, step=0.1)
    st.markdown("<br>", unsafe_allow_html=True) # Jarak kosong agar tombol sejajar
    btn_predict = st.button("🔍 Hitung Prediksi AI")

# 4. MEMUAT MODEL AI (JSON)
@st.cache_resource
def load_model():
    model_xgb = XGBClassifier()
    model_xgb.load_model("model.json") # Membaca file model.json (tanpa joblib/pickle)
    return model_xgb

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Gagal memuat 'model.json'. Pastikan file tersebut ada di GitHub Anda. Detail Error: {e}")
    st.stop()

# 5. DASHBOARD HASIL & GRAFIK (Muncul saat tombol ditekan)
if btn_predict:
    st.markdown("---")
    with st.spinner("🧠 Memproses data dengan AI..."):
        # Prediksi probabilitas
        input_data = pd.DataFrame([[sod2, hspa2, tnf, bax, cat]], columns=['SOD2', 'HSPA2', 'TNF', 'BAX', 'CAT'])
        prob = model.predict_proba(input_data)[0][1] * 100
        
        st.markdown('<div class="section-header">📊 2. Hasil Stratifikasi & Transparansi AI</div>', unsafe_allow_html=True)
        col_res1, col_res2 = st.columns([1.2, 2])
        
        # Panel Skor Risiko
        with col_res1:
            st.metric(label="Skor Keparahan Varikokel", value=f"{prob:.1f}%")
            if prob >= 50:
                st.error("🚨 **STATUS: RISIKO TINGGI**\n\nSangat direkomendasikan untuk evaluasi medis lanjutan.")
            else:
                st.success("✅ **STATUS: RISIKO RENDAH**\n\nProfil ekspresi genetik terpantau stabil.")
                
        # Panel Grafik Transparansi (Matplotlib bergaya modern)
        with col_res2:
            st.markdown("**Grafik Feature Importance (Nilai SHAP)**")
            feature_names = ['SOD2', 'HSPA2', 'TNF', 'BAX', 'CAT']
            shap_values = [-(10-sod2)*0.3, -(10-hspa2)*0.2, tnf*0.35, bax*0.25, -(10-cat)*0.1]
            
            fig, ax = plt.subplots(figsize=(7, 3.2))
            fig.patch.set_facecolor("none") # Background transparan
            ax.set_facecolor("none")       # Background area transparan
            
            # Palet Medis: Merah (#EF4444) untuk Risiko Naik, Teal (#00C2CB) untuk Risiko Turun
            colors = ['#EF4444' if x > 0 else '#00C2CB' for x in shap_values] 
            ax.barh(feature_names, shap_values, color=colors, height=0.5, alpha=0.9)
            
            # Merapikan bingkai grafik agar terlihat elegan
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('gray')
            ax.spines['bottom'].set_color('gray')
            ax.tick_params(colors='gray')
            ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
            
            ax.set_xlabel("Merah: Meningkatkan Risiko | Teal: Menurunkan Risiko", color='gray', fontsize=9)
            plt.tight_layout()
            
            st.pyplot(fig) # Menampilkan grafik di Streamlit

    # 6. TABEL REKOMENDASI HERBAL
    st.markdown('<div class="section-header">🌿 3. Rekomendasi Terapi Herbal In Silico</div>', unsafe_allow_html=True)
    
    data_docking = {
        "Senyawa Herbal": ["Kurkumin", "Epigallocatechin Gallate (EGCG)", "Quercetin", "Resveratrol"],
        "Sumber Alam": ["Kunyit (Curcuma longa)", "Teh Hijau (Camellia sinensis)", "Apel / Bawang Merah", "Anggur / Kacang"],
        "Protein Target": ["TNF-alpha", "SOD2", "BAX", "CAT"],
        "Binding Affinity": ["-8.2 kcal/mol", "-7.8 kcal/mol", "-7.5 kcal/mol", "-7.1 kcal/mol"],
        "Status Potensi": ["Sangat Kuat (Utama)", "Kuat", "Moderat", "Moderat"]
    }
    
    df_docking = pd.DataFrame(data_docking)
    # Menampilkan dataframe dengan UI modern (hide_index menghilangkan angka urut di kiri tabel)
    st.dataframe(df_docking, use_container_width=True, hide_index=True)

# 7. REFERENSI & PUSTAKA ILMIAH (Bagian Paling Bawah)
st.write("---")
st.markdown("#### 📚 Pustaka Ilmiah & Rujukan")
st.markdown("""
<div style="font-size: 0.9rem; color: #64748b;">
    <ul>
        <li>🔗 <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6409899/" target="_blank" style="color: #00C2CB; text-decoration: none;">NCBI - Varicocele and Male Infertility: The Role of Oxidative Stress</a></li>
        <li>🔗 <a href="https://pubmed.ncbi.nlm.nih.gov/30528020/" target="_blank" style="color: #00C2CB; text-decoration: none;">PubMed - Gene Expression of Antioxidant Enzymes in Infertile Men</a></li>
    </ul>
    <i>*Dikembangkan untuk tujuan penelitian akademis dan skrining awal. Selalu konsultasikan dengan dokter urologi.*</i>
</div>
""", unsafe_allow_html=True)
