import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static
import plotly.express as px

# Import data preparation
try:
    from data_preparation_jawa_timur import JAWA_TIMUR_DAERAH, fetch_air_quality
    IMPORT_SUCCESS = True
except ImportError as e:
    st.error(f"Error import: {e}")
    IMPORT_SUCCESS = False

# Load model dan mapping
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_jawa_timur.pkl')
        preprocessor = joblib.load('preprocessor_jawa_timur.pkl')
        category_mapping = joblib.load('category_mapping.pkl')
        return model, preprocessor, category_mapping
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Fungsi predict
def predict_quality(features, model, preprocessor, category_mapping):
    """Prediksi kualitas udara"""
    df_features = pd.DataFrame([features])
    X_processed = preprocessor.transform(df_features)
    
    prediction = model.predict(X_processed)[0]
    probabilities = model.predict_proba(X_processed)[0]
    
    reverse_mapping = {v: k for k, v in category_mapping.items()}
    category = reverse_mapping[prediction]
    confidence = max(probabilities)
    
    prob_dict = {}
    for i, prob in enumerate(probabilities):
        category_name = reverse_mapping[i]
        prob_dict[category_name] = prob
    
    return category, confidence, prob_dict

# Konfigurasi app
st.set_page_config(
    page_title="Kualitas Udara Jawa Timur",
    page_icon="🌫️",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
    .good { background-color: #d4edda; padding: 10px; border-radius: 5px; }
    .moderate { background-color: #fff3cd; padding: 10px; border-radius: 5px; }
    .unhealthy { background-color: #f8d7da; padding: 10px; border-radius: 5px; }
    .header { color: #1f77b4; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="header">🌫️ Monitoring Kualitas Udara Jawa Timur</h1>', unsafe_allow_html=True)

# Load model
model, preprocessor, category_mapping = load_model()
MODEL_READY = model is not None and preprocessor is not None

# Sidebar
st.sidebar.header("Pilih Daerah")
daerah = st.sidebar.selectbox("Kabupaten/Kota", list(JAWA_TIMUR_DAERAH.keys()))

if st.sidebar.button("Ambil Data"):
    if not IMPORT_SUCCESS:
        st.error("Tidak dapat mengimpor modul data")
    else:
        with st.spinner("Mengambil data..."):
            try:
                data = fetch_air_quality(daerah)
                st.session_state.current_data = data
                st.sidebar.success("Data berhasil diambil!")
            except Exception as e:
                st.error(f"Error: {e}")

# Main content
if 'current_data' in st.session_state:
    data = st.session_state.current_data
    daerah_info = JAWA_TIMUR_DAERAH[daerah]
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"Data Kualitas Udara - {daerah}")
        
        # Metrics
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            st.metric("PM2.5", f"{data['PM2.5']} μg/m³")
            st.metric("Suhu", f"{data['Suhu']}°C")
        with col1b:
            st.metric("PM10", f"{data['PM10']} μg/m³")
            st.metric("Kelembaban", f"{data['Kelembaban']}%")
        with col1c:
            st.metric("CO", f"{data['CO']} mg/m³")
            st.metric("Tipe Wilayah", daerah_info['tipe'])
        
        # Chart
        chart_data = {
            'Parameter': ['PM2.5', 'PM10', 'CO'],
            'Nilai': [data['PM2.5'], data['PM10'], data['CO']]
        }
        fig = px.bar(chart_data, x='Parameter', y='Nilai', 
                    title=f"Parameter Kualitas Udara - {daerah}",
                    color='Parameter')
        st.plotly_chart(fig)
    
    with col2:
        st.header("Hasil Analisis")
        
        if MODEL_READY:
            try:
                category, confidence, probabilities = predict_quality(
                    data, model, preprocessor, category_mapping
                )
                
                # Tampilkan kategori
                if category == 'Sangat Baik':
                    st.markdown('<div class="good">', unsafe_allow_html=True)
                    st.success("✅ SANGAT BAIK")
                elif category == 'Baik':
                    st.markdown('<div class="good">', unsafe_allow_html=True)
                    st.info("👍 BAIK")
                elif category == 'Sedang':
                    st.markdown('<div class="moderate">', unsafe_allow_html=True)
                    st.warning("⚠️ SEDANG")
                else:
                    st.markdown('<div class="unhealthy">', unsafe_allow_html=True)
                    st.error("❌ TIDAK SEHAT")
                
                st.write(f"**Kategori:** {category}")
                st.write(f"**Confidence:** {confidence:.1%}")
                
                # Probabilities
                st.write("**Probabilitas:**")
                for cat, prob in probabilities.items():
                    st.write(f"- {cat}: {prob:.1%}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Rekomendasi
                st.subheader("💡 Rekomendasi")
                if category in ['Sangat Baik', 'Baik']:
                    st.success("""
                    ✅ **SEMUA AKTIVITAS AMAN**
                    - Olahraga luar ruangan bebas dilakukan
                    - Bersepeda, jogging, bermain di taman, piknik aman
                    - Tidak perlu pembatasan aktivitas
                    - Cocok juga untuk anak-anak, lansia, dan penderita penyakit pernapasan.
                    """)
                elif category == 'Sedang':
                    st.warning("""
                    ⚠️ **HATI-HATI**
                    - Aktivitas ringan-sedang masih boleh dilakukan
                    - Jalan santai, bersepeda ringan, atau olahraga pagi/sore
                    - Hindari aktivitas berat berkepanjangan
                    - Kelompok rentan (anak-anak, lansia, dan penderita asma/penyakit paru) sebaiknya di dalam ruangan
                    - Gunakan masker jika harus keluar
                    """)
                else:
                    st.error("""
                    ❌ **HINDARI AKTIVITAS LUAR**
                    - Semua orang sebaiknya di dalam ruangan
                    - Gunakan air purifier jika memungkinkan
                    - Tunda aktivitas luar yang tidak penting
                    - Jika terpaksa, gunakan pelindung pernapasan yang sesuai dan batasi waktu di luar seminimal mungkin
                    """)
                    
            except Exception as e:
                st.error(f"Error prediksi: {e}")
        else:
            st.warning("Model belum tersedia")
        
        # Peta
        st.subheader("🗺️ Peta")
        m = folium.Map(location=[daerah_info['lat'], daerah_info['lon']], zoom_start=10)
        folium.Marker(
            [daerah_info['lat'], daerah_info['lon']],
            popup=f"{daerah}",
            tooltip="Klik untuk detail"
        ).add_to(m)
        folium_static(m, width=300)

else:
    st.info("Pilih daerah dan klik 'Ambil Data' untuk memulai")
    
    if MODEL_READY:
        st.success("✅ Model siap digunakan!")
    else:
        st.warning("⚠️ Jalankan training model terlebih dahulu")

# Footer
st.markdown("---")
st.markdown("**Sistem Monitoring Kualitas Udara Jawa Timur**")