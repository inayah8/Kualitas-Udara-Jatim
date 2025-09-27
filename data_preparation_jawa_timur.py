import pandas as pd
import numpy as np
import random
from datetime import datetime

# Daftar kabupaten/kota di Jawa Timur
JAWA_TIMUR_DAERAH = {
    # Kota Industri - Polusi tinggi
    'Surabaya': {'lat': -7.2575, 'lon': 112.7521, 'tipe': 'industri', 'faktor_polusi': 1.3},
    'Gresik': {'lat': -7.150, 'lon': 112.650, 'tipe': 'industri', 'faktor_polusi': 1.4},
    'Sidoarjo': {'lat': -7.450, 'lon': 112.720, 'tipe': 'industri', 'faktor_polusi': 1.3},
    
    # Kota Besar - Polusi sedang
    'Malang': {'lat': -7.980, 'lon': 112.630, 'tipe': 'kota_besar', 'faktor_polusi': 1.1},
    'Kediri': {'lat': -7.820, 'lon': 112.010, 'tipe': 'kota_besar', 'faktor_polusi': 1.0},
    'Madiun': {'lat': -7.630, 'lon': 111.520, 'tipe': 'kota_besar', 'faktor_polusi': 1.0},
    
    # Pantai - Polusi rendah
    'Banyuwangi': {'lat': -8.210, 'lon': 114.370, 'tipe': 'pantai', 'faktor_polusi': 0.7},
    'Pasuruan': {'lat': -7.640, 'lon': 112.910, 'tipe': 'pantai', 'faktor_polusi': 0.8},
    'Probolinggo': {'lat': -7.750, 'lon': 113.210, 'tipe': 'pantai', 'faktor_polusi': 0.8},
    
    # Pegunungan - Udara bersih
    'Batu': {'lat': -7.870, 'lon': 112.530, 'tipe': 'pegunungan', 'faktor_polusi': 0.6},
    'Lumajang': {'lat': -8.180, 'lon': 113.220, 'tipe': 'pegunungan', 'faktor_polusi': 0.7},
    'Trenggalek': {'lat': -8.070, 'lon': 111.720, 'tipe': 'pegunungan', 'faktor_polusi': 0.6},
    
    # Lainnya
    'Jember': {'lat': -8.170, 'lon': 113.710, 'tipe': 'campuran', 'faktor_polusi': 0.9},
    'Bojonegoro': {'lat': -7.480, 'lon': 111.390, 'tipe': 'campuran', 'faktor_polusi': 0.9},
    'Tuban': {'lat': -6.880, 'lon': 112.060, 'tipe': 'campuran', 'faktor_polusi': 0.9},
}

def generate_air_quality(daerah, faktor_polusi, tipe):
    """Generate data kualitas udara yang realistis"""
    
    if tipe == 'industri':
        pm25 = random.uniform(30, 80)
        pm10 = random.uniform(60, 150)
        co = random.uniform(10, 25)
    elif tipe == 'kota_besar':
        pm25 = random.uniform(20, 50)
        pm10 = random.uniform(40, 100)
        co = random.uniform(6, 15)
    elif tipe == 'pantai':
        pm25 = random.uniform(10, 30)
        pm10 = random.uniform(20, 60)
        co = random.uniform(3, 8)
    else:  # pegunungan
        pm25 = random.uniform(5, 20)
        pm10 = random.uniform(10, 40)
        co = random.uniform(2, 6)
    
    # Apply faktor polusi
    pm25 *= faktor_polusi
    pm10 *= faktor_polusi
    co *= faktor_polusi
    
    # Data cuaca
    suhu = random.uniform(25, 32)
    kelembaban = random.uniform(65, 85)
    
    return {
        'PM2.5': round(pm25, 1),
        'PM10': round(pm10, 1),
        'CO': round(co, 1),
        'Suhu': round(suhu, 1),
        'Kelembaban': round(kelembaban, 1),
        'Daerah': daerah
    }

def fetch_air_quality(daerah):
    """Fetch data untuk daerah tertentu"""
    if daerah not in JAWA_TIMUR_DAERAH:
        # Default data jika daerah tidak ditemukan
        return {
            'PM2.5': 25.0, 'PM10': 50.0, 'CO': 8.0, 
            'Suhu': 28.0, 'Kelembaban': 75.0, 'Daerah': daerah
        }
    
    daerah_info = JAWA_TIMUR_DAERAH[daerah]
    return generate_air_quality(daerah, daerah_info['faktor_polusi'], daerah_info['tipe'])

def determine_category(pm25, pm10, co):
    """Tentukan kategori kualitas udara"""
    if pm25 <= 15.5 and pm10 <= 50 and co <= 4:
        return 'Sangat Baik'
    elif pm25 <= 55.4 and pm10 <= 150 and co <= 9:
        return 'Baik'
    elif pm25 <= 150.4 and pm10 <= 350 and co <= 15:
        return 'Sedang'
    elif pm25 <= 250.4 and pm10 <= 420 and co <= 30:
        return 'Tidak Sehat'
    else:
        return 'Sangat Tidak Sehat'

def create_dataset(n_samples=2000):
    """Buat dataset untuk training"""
    daerah_list = list(JAWA_TIMUR_DAERAH.keys())
    
    data = []
    for _ in range(n_samples):
        daerah = random.choice(daerah_list)
        daerah_info = JAWA_TIMUR_DAERAH[daerah]
        
        air_data = generate_air_quality(daerah, daerah_info['faktor_polusi'], daerah_info['tipe'])
        category = determine_category(air_data['PM2.5'], air_data['PM10'], air_data['CO'])
        
        air_data['Kategori'] = category
        data.append(air_data)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_dataset()
    print("Dataset sample:")
    print(df.head())
    print("\nKategori distribution:")
    print(df['Kategori'].value_counts())