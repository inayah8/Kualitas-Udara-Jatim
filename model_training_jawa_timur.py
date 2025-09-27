import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
import joblib

from data_preparation_jawa_timur import create_dataset

# Mapping kategori yang FIX
CATEGORY_MAPPING = {
    'Sangat Baik': 0,
    'Baik': 1, 
    'Sedang': 2,
    'Tidak Sehat': 3,
    'Sangat Tidak Sehat': 4
}

def prepare_data():
    """Siapkan data untuk training - return HANYA 3 values"""
    df = create_dataset(n_samples=2000)
    
    # Features dan target
    X = df[['PM2.5', 'PM10', 'CO', 'Suhu', 'Kelembaban', 'Daerah']]
    y = df['Kategori'].map(CATEGORY_MAPPING)
    
    # Preprocessor
    numeric_features = ['PM2.5', 'PM10', 'CO', 'Suhu', 'Kelembaban']
    categorical_features = ['Daerah']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
    
    # Transform data
    X_processed = preprocessor.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    # Return HANYA 3 values
    return X_train, X_test, y_train, y_test, preprocessor

def train_model():
    """Train model dan return HANYA 2 values"""
    print("Memulai training model...")
    
    # Dapatkan data yang sudah diproses
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=list(CATEGORY_MAPPING.keys())))
    
    # Save model dan preprocessor
    joblib.dump(model, 'model_jawa_timur.pkl')
    joblib.dump(preprocessor, 'preprocessor_jawa_timur.pkl')
    joblib.dump(CATEGORY_MAPPING, 'category_mapping.pkl')
    
    print("Model berhasil disimpan!")
    return model, preprocessor  # HANYA return 2 values

def predict_quality(features, model, preprocessor):
    """Prediksi kualitas udara"""
    # Buat DataFrame dari features
    df_features = pd.DataFrame([features])
    
    # Preprocess features
    X_processed = preprocessor.transform(df_features)
    
    # Predict
    prediction = model.predict(X_processed)[0]
    probabilities = model.predict_proba(X_processed)[0]
    
    # Convert ke kategori
    reverse_mapping = {v: k for k, v in CATEGORY_MAPPING.items()}
    category = reverse_mapping[prediction]
    confidence = max(probabilities)
    
    # Probabilitas semua kategori
    prob_dict = {}
    for i, prob in enumerate(probabilities):
        category_name = reverse_mapping[i]
        prob_dict[category_name] = prob
    
    return category, confidence, prob_dict

if __name__ == "__main__":
    # Train model
    model, preprocessor = train_model()  # Hanya terima 2 values
    
    # Test prediction
    test_features = {
        'PM2.5': 35.0,
        'PM10': 70.0, 
        'CO': 12.0,
        'Suhu': 28.0,
        'Kelembaban': 75.0,
        'Daerah': 'Surabaya'
    }
    
    category, confidence, probabilities = predict_quality(test_features, model, preprocessor)
    
    print(f"\nTest Prediction: {category} (Confidence: {confidence:.3f})")
    print("Probabilities:")
    for cat, prob in probabilities.items():
        print(f"  {cat}: {prob:.3f}")