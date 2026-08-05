import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from data_loader import load_data
from elo import compute_elo
from features import create_features

# Step 1: Load and prepare data
print("📊 Loading data...")
df = load_data()

print("⚡ Computing Elo ratings...")
df = compute_elo(df)

print("🔧 Creating features...")
df = create_features(df)  # This already filters to post-2022

# Step 2: Define features and target
feature_cols = [
    'elo_diff',
    'elo_abs_diff',
    'avg_elo',
    'is_friendly',
    'is_neutral',
    'home_form',      # ✅ NEW
    'away_form',      # ✅ NEW
    'home_gd_avg',    # ✅ NEW
    'away_gd_avg'     # ✅ NEW
]
X = df[feature_cols]
y = df['result']

print(f"\n📊 Training data shape: {X.shape}")
print(f"📊 Feature columns: {feature_cols}")

# Step 3: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# Step 4: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train XGBoost
print("\n🚀 Training XGBoost classifier...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))

# Step 7: Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Feature Importance:")
print(importance)

import joblib

# Save the model, scaler, and feature columns
joblib.dump(model, 'models/xgboost_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl')

print("\n✅ Model saved to 'models/' folder.")